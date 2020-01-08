# -*- coding: utf8 -*-
#   Copyright 2019 JSALT2019 Distant Supervision Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import copy
import io
import logging
import os
import sys
import re
import zipfile

import torch.utils.data
import PIL.Image
import torchvision
import numpy as np
import pandas as pd

# Future FIXME: extract Path from Aligner into its own class and just import class Path
import distsup
from distsup import aligner

import egs.scribblelens.utils
from distsup.alphabet import Alphabet
from distsup.utils import construct_from_kwargs


class ScribbleLensDataset(torch.utils.data.Dataset):
    """Scribble Lens dataset."""

    def __init__(self,
                 root='data/scribblelens.corpus.v1.zip',
                 dataframe_filename='data/scribblelens.corpus.v1.pkl',
                 alignment_root="", # Default empty i.e. unused
                 split=None,
                 slice=None, # tasman, kieft, brouwers
                 slice_query=None,
                 slice_filename=None,
                 colormode='bw',
                 vocabulary="", # The alphabet filename in json format
                 vocabulary_query=None,
                 write_vocabulary=False,
                 transcript_mode=2,
                 target_height=32,
                 target_width=-1,
                 transform=None
                 ):
        """
        Args:
            root (string): Root directory of the dataset.
            alignmentRoot (string): Root directory of the path alignments. There should be one .ali file per image.
            split (string): The subset of data to provide.
                Choices are: train, test, supervised, unsupervised.
            slice_filename (string): Don't use existing slice and use a custom slice from a filename. The file
                should use the same format as in the dataset.
            colormode (string): The color of data to provide.
                Choices are: bw, color, gray.
            alphabet (dictionary): Pass in a pre-build alphabet from external source, or build during training if empty
            transcript_mode(int): Defines how we process space in target text, and blanks in targets [1..5]
            target_height (int, None): The height in pixels to which to resize the images.
                Use None for the original size, -1 for proportional scaling
            target_width (int, None): The width in pixels to which to resize the images.
                Use None for the original size, -1 for proportional scaling
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Note:
            The alphabet or vocabulary needs to be generated with an extra tool like
                     generateAlphabet.py egs/scribblelens/yamls/tasman.yaml
        """
        self.root = root
        self.write_vocabulary = write_vocabulary
        self.vocabulary_query = vocabulary_query

        self.file = zipfile.ZipFile(root)

        self.target_width = target_width
        self.target_height = target_height
        if transform:
            self.pre_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                construct_from_kwargs(transform, additional_parameters={'scribblelens': True}),
                                torchvision.transforms.ToPILImage(),
                                ])
        else:
            self.pre_transform = None
        
        self.transform = torchvision.transforms.Compose([
                 torchvision.transforms.Grayscale(),
                 torchvision.transforms.ToTensor(),
                 ])

        logging.debug(f"ScribbleLensDataset() constructor for split = {split}")

        self.dataframe_filename = dataframe_filename
        df = pd.read_pickle(self.dataframe_filename)
        df['alignment'] = np.nan
        df['alignment_rle'] = np.nan
        df['alignment_text'] = np.nan
        df['text'] = np.nan

        df['alignment'] = df['alignment'].astype(object)
        df['alignment_rle'] = df['alignment_rle'].astype(object)
        df['alignment_text'] = df['alignment_text'].astype(object)
        df['text'] = df['text'].astype(object)

        # 'vocabulary' Filename from .yaml. alphabet has the vocabulary as a dictionary for CTC output targets.
        self.transcriptMode = transcript_mode
        assert (1 <= self.transcriptMode <= 5)

        self.vocabulary = vocabulary
        if self.vocabulary != "" and not os.path.isfile(self.vocabulary):
            logging.error(f"You specified a vocabulary that does not exist: {self.vocabulary}")
            sys.exit(4)

        self.alphabet = Alphabet(self.vocabulary)
        self.vocab_size = len(self.alphabet)
        self.must_create_alphabet = ((self.vocabulary == '') or self.write_vocabulary)
        self.nLines = 0

        authors = {'tasman', 'zeewijck', 'brouwer.chili', 'craen.de.vos.ijszee',
                   'van.neck.tweede', 'van.neck.vierde', 'kieft'}

        assert(colormode in {'bw', 'color', 'gray'})
        assert(split is None or split in {'all', 'train', 'test', 'supervised', 'unsupervised'})
        assert(slice is None or slice in ({'empty', 'query', 'custom'} | authors))
        assert target_height != -1 or target_width != -1

        self.slice = slice
        self.split = split

        if self.must_create_alphabet:
            if self.vocabulary_query is None:
                self.vocabulary_query = "split == 'train'"
                logging.warning(f'The vocabulary query has not been set. Setting it to "{self.vocabulary_query}"')

            # Build the alphabet using the training data only
            self.build_alphabet(df.query(self.vocabulary_query), self.alphabet, transcript_mode)

        # Select the data

        if slice_filename is not None and self.slice != 'custom':
            logging.error(f'Slice filename set to "{slice_filename}" '
                          f'yet slice is not "custom" but "{self.slice}".')
            sys.exit(1)

        if slice_query is not None and self.slice != 'query':
            logging.error(f'Slice query set to "{slice_filename}" '
                          f'yet slice is not "query" but "{self.slice}".')
            sys.exit(1)

        df_selection = df
        if self.slice == 'query':
            # Select with query
            df_selection = df.query(slice_query)

        elif self.slice == 'custom':
            assert slice_filename is not None
            # Select with query
            with open(slice_filename) as f:
                img_filenames = [l.strip().split()[0] for l in f.read().split() if l.strip()]

            df_selection = df[df['img_filename'].isin(set(img_filenames))]

        elif self.slice in authors:
            df_selection = df[df['scribe'] == self.slice]

        elif self.slice is None:
            pass

        else:
            raise ValueError(f'Slice "{self.slice}" not available')

        if self.split is not None:
            if self.split == 'supervised':
                df_selection = df_selection[df_selection['transcribed']]

            elif self.split == 'unsupervised':
                df_selection = df_selection[~df_selection['transcribed']]

            elif self.split in {'train', 'test'}:
                df_selection = df_selection[df_selection['split'] == self.split]

            else:
                raise ValueError(f'Split "{self.split}" not available.')

        self.get_transcriptions(df_selection, self.alphabet, transcript_mode)

        self.get_alignments(df_selection, alignment_root)

        self.metadata = {
            'alignment': {
                'type': 'categorical',
                'num_categories': len(self.alphabet)},
            'text': {
                'type': 'categorical',
                'num_categories': len(self.alphabet)},
        }

        self.file.close()
        self.file = None

        self.df = df_selection

    def build_alphabet(self, df, alphabet, transcript_mode):
        for index, row in df.iterrows():
            text = self.read_transcript(row['text_filename'],
                                        alphabet,
                                        transcript_mode,
                                        build_alphabet=True)
            df.at[index, 'text'] = torch.tensor(text)

        if self.write_vocabulary:
            logging.warning(f'(Over)writing the vocabulary to {self.vocabulary}.')
            egs.scribblelens.utils.writeDictionary(self.vocabulary)

    def get_transcriptions(self, df, alphabet, transcript_mode):
        for index, row in df[df['text_filename'].notnull()].iterrows():
            text = self.read_transcript(row['text_filename'],
                                        alphabet,
                                        transcript_mode,
                                        build_alphabet=False)
            df.at[index, 'text'] = torch.tensor(text)

    def get_alignments(self, df, alignment_root):
        self.alignmentFile = None  # Optional
        if alignment_root != "":
            self.alignmentFile = zipfile.ZipFile(alignment_root)
            self.pathAligner = distsup.aligner.Aligner("none", self.alphabet)  # Path I/O

        for index, row in df.iterrows():
            # Set alignment field in item[]
            if self.alignmentFile is not None:
                aliFilename = row['img_filename'].replace(".jpg",".ali")
                classIndexAlignmentList, myTargetStr = self.pathAligner.readAlignmentFileFromZip(aliFilename,
                                                                                                 self.alignmentFile)
                alignment = torch.IntTensor(classIndexAlignmentList)
                df.at[index, 'alignment'] = alignment
                df.at[index, 'alignment_text'] = myTargetStr
                df.at[index, 'alignment_rle'], _ = distsup.utils.rleEncode(alignment)

        if self.alignmentFile is not None:
            self.alignmentFile.close()

    '''
        Input:
            Aligner class instance. This could be a "recognition path aligner" or a "forced aligner"

            if targets_ == None (and targets_len_ == None) then we are in recognition/test/forward-only mode.
    '''
    def decode(self, aligner_, \
        decodesWithBlanks_, decodes_, \
        log_probs_, log_prob_lens_, \
        targets_, targets_len_, \
        batch_, \
        verbose_=0 ):

        featuresLen = batch_['features_len']
        imageFilenameList_ = batch_['img_filename']
        try:
            # Expect original image size of [ nRows x nColumns ] == [orgHeight x orgWidth]
            orgImageSizes_ = batch_['img_size']
        except:
            orgImageSizes_ = None

        batchSize = log_probs_.shape[1]

        assert len(log_probs_.shape) == 3
        assert len(decodesWithBlanks_) == len(imageFilenameList_), "ERROR: batchSize for decoded path should be same a list of filenames!"

        # Decode the batch
        for i in range(0,batchSize):
            sz = len(decodesWithBlanks_[i])
            assert log_prob_lens_[i] == sz

            currentPathWithBlanks = decodesWithBlanks_[i]
            currentLogProbs = log_probs_[:sz,i,:]
            currentPathNoBlanks = torch.IntTensor(decodes_[i])
            if targets_ is not None:
                currentTargets = targets_[i]
                currentTargetLen = targets_len_[i]
            else:
                currentTargets = None
                currentTargetLen = None

            if verbose_ > 0:
                # processPaths() knows how to handle empty targets/targetlen
                self.processPaths( currentPathWithBlanks, currentPathNoBlanks, \
                    currentTargets, currentTargetLen, \
                    i, self.alphabet, verbose_ )

            # Write one path to file, either via forced alignment or based on this recognized alignment.
            if aligner_ != None:
                orgHeight = orgImageSizes_[i][0].item()
                orgWidth = orgImageSizes_[i][1].item()
                resizedHeight = batch_['features'].shape[2]
                nFeatures = resizedWidth = featuresLen[i].item() # FIX --> This was wrong earlier: batch_['features'].shape[1]

                assert orgHeight > 0
                assert resizedHeight > 0

                '''
                Note: we have 3 (stretch) factors, and call makePath in Aligner or ForcedAligner
                (1) the org vs resized (32 pixels) e.g. factor 5.75 (stretchFactor1)
                (2) the compression of the CNN encoder e.g. a factor of 6 or 8 (stretchFactor2)
                (3) The difference between nFeature on input and logProbs CTC outputs

                stretchFactor1 = (orgHeight * 1.0) / (resizedHeight * 1.0)
                stretchFactor2 = orgWidth / (len(currentLogProbs)  * 1.0)
                stretchFactor3 = resizedWidth / (len(currentLogProbs) * 1.0)

                FIXME: add an assert that when (currentTargets == None) we are working with an Aligner, and not an ForcedAligner!
                '''
                pathStr = aligner_.makePath(currentPathWithBlanks, currentLogProbs,
                    currentTargets, currentTargetLen,
                    self.alphabet,
                    imageFilenameList_[i], orgImageSizes_[i],
                    nFeatures ) # Target length for path == nFeatures. Guaranteed the same length

    '''
        We get 1 path which has no symbols removed, one path with no reps, no blanks, and a target path
        All are tensors [ 1 x something]
        verbose is set via .yaml, model
        If targets_ == None then we are in recognition forward() mode, also assume that targetLen_ == None
        and just process the recognized strings, not the targets
    '''
    def processPaths(self, pathWithBlanks_, pathClean_, targets_, targetLen_, idx_, alphabet_, verbose_ = 0):

        if targets_ is not None:
            mytarget = targets_[:targetLen_]
            assert mytarget.argmin() >= 0
            assert mytarget[ mytarget.argmax() ] < len(alphabet_)

        prefix = "ProcessPaths(" + str(idx_) + ")"
        if verbose_ > 1:
            print ( prefix + " : path with all blanks " + str(pathWithBlanks_))
        if verbose_ > 0:
            print ( prefix + " : path with  no blanks, no repetitions " + str(pathClean_))

        pathStrWithBlanks = alphabet_.idx2str(pathWithBlanks_)
        pathStrClean      = alphabet_.idx2str(pathClean_)
        if targets_ is not None:
            targetStr         = alphabet_.idx2str(mytarget)
            targetStrClean    = alphabet_.idx2str(mytarget, noBlanks=True)
            if verbose_ > 0:
                print ( prefix + " : targets_ " + str(mytarget))

        if verbose_ > 1:
            print ( prefix + " : pathStrWithBlanks :" + str(pathStrWithBlanks))
        if verbose_ > 0:
            print ( prefix + " : pathStrClean      :" + str(pathStrClean))
        if verbose_ > 1 and targets_ is not None:
            print ( prefix + " : targetStr         :" + str(targetStr))
        if verbose_ > 0 and targets_ is not None:
            print ( prefix + " : targetStrClean    :" + str(targetStrClean))

    '''
     buildAlphabetMode should be false in test/validation mode, and true in training

     mode 1 means: remove all spaces in input text, and just put 1 blank at SENT START/END
     mode 2 is roughly "remove all spaces in input text, and use blanks BETWEEN all characters as targets"
     mode 3: no blanks, no spaces, no word boundaries encoded - plain character seq
     mode 4 keep word boundaries (spaces) and add an extra blank before and after the sentence
     mode 5 as mode 4, keep word boundaries, but no extra blanks

     Default is mode 2, but mode 4 or 5 is nice for user.
     Alphabet is a reference to dictionary i.e. just manipulate, and do not assign

     Special symbol:
        '*' id==0 which is the 'blank' in CTC
        '_' id==1 which is the 'SPACE' between words in output strings
        Unknown symbols are mapped to blanks if necessary.
    '''

    @staticmethod
    def process_transcript_line(line, alphabet, mode=2, build_alphabet=False):
        # count lines
        idx = 0

        # Build a python list of encoded characters, via alphabet map/dict
        tmp = []

        # Step #0 : deal with alphabet from training text
        if build_alphabet and len(alphabet) == 0:
            alphabet.insertDict('*', 0)  # symbol and idx in Neural Network
            if mode == 4 or mode == 5:
                alphabet.insertDict(' ', 1)

        # Step #1: make input characters from line_ to a python lists tmp with encoded chars
        # Note: the "+1" to address the fact that class 0 is gap/silence in CTC
        i = 0
        for c in line:
            if mode < 4:
                if c != ' ' and c != '.' and c != '\'':  # Skip these noisy ones
                    if build_alphabet and not alphabet.existDict(c):
                        alphabet.insertDict(c, len(alphabet))

                    # If validation/recognition/test mode, ignore unknown symbols we do not know. Replace with blank.
                    if not build_alphabet and not alphabet.existDict(c):
                        c = "*"
                    tmp.append(alphabet.ch2idx(c))

                    i = i + 1

            else:  # mode 4 or 5 , keep word boundaries
                if c != '.' and c != '\'':  # Skip these noisy ones
                    if not alphabet.existDict(c):
                        if build_alphabet:
                            alphabet.insertDict(c, len(alphabet))
                        else:
                            # Not building a dictionary here.
                            # We are in recognition/eval/test mode. But, we have an unknown.
                            logging.debug("processTranscriptLine() the symbol :" + str(
                                c) + ": is an UNK in this mode and alphabet. Ignore.")
                            c = '*'  # Replace unknown symbol with existing symbol in alphabet. Use a blank '*'
                    tmp.append(alphabet.ch2idx(c))

                    i = i + 1

        logging.debug("mode = " + str(mode) + ", tmp list =" + str(tmp))
        logging.debug("tmp list len =" + str(len(tmp)))

        # Step #2: transfer encoded lists 'tmp' to 'result' plus SENTENCE START/END symbols
        result = []

        if mode == 1:  # mode (a) '01230'
            result.append(0)
            for i in tmp:
                result.append(i)
            result.append(0)

        if mode == 2:  # mode (b) '0102030' *a*m*o*r*d*e*d*e*x*
            result.append(0)
            for i in tmp:
                result.append(i)
                result.append(0)

        if mode == 3:  # mode (c), no '0' labels, '123' amordede...
            result = [i for i in tmp]

        # mode (d) '012130' i.e. insert space/underscore id==1
        # at word boundaries in transcripts *amor_de_dexar_las_cos*
        if mode == 4:
            result.append(0)
            for i in tmp:
                result.append(i)
            result.append(0)  # add extra 0 at word end

        # mode (e) '1213' i.e. insert word boundaries id=1=='_' at word boundaries in transcripts amor_de_dexar_las_cos
        # and no trailing/leading blank
        if mode == 5:
            for i in tmp:
                result.append(i)

        idx = idx + 1

        assert idx == 1  # 1 line only

        return result

    # Read a .txt transcript file with > one <  line of text
    # Assume one line of text in transcript file.
    def read_transcript(self, filename, alphabet, transcript_mode, build_alphabet=False):
        line_count = 0
        with self.file.open(filename) as f:
            raw_text = f.read().decode('utf8').strip()  # Read Unicode characters

            # self.Alphabet is a ref arg if you like, lives inside this data class
            # It is either 0 length if new, or filled earlier from vocabulary file.
            result = ScribbleLensDataset.process_transcript_line(raw_text,
                                                                 alphabet,
                                                                 mode=transcript_mode,
                                                                 build_alphabet=build_alphabet)
            assert len(result) > 0
            line_count += 1

        assert line_count == 1
        logging.debug(f"ScribbleLensDataset:readTranscript() result ::{result}::")

        return result

    # Return a flag 'True' when processing a training dataset for a training run,
    # and False when doing recognitions, path alignments.
    def getTrainingMode(self):
        return self.trainingMode

    def __del__(self):
        if self.file:
            self.file.close()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self.file:  # Reopen to work with multiprocessing
            self.file = zipfile.ZipFile(self.root)

        df_item = self.df.iloc[idx]
        item = df_item.to_dict()

        item['scribe'] = df_item['scribe']
        item['scribe_id'] = df_item['scribe_id']

        new_item = copy.deepcopy(item)

        # Load and transform the image. Q: Why not mode 'L' for Gray level?
        with self.file.open(item['img_filename']) as img_f:
            image = PIL.Image.open(img_f).convert('RGB')

        if self.pre_transform:
            image = self.pre_transform(image)

        # Pass down the original image size, e.g., to stretch the forced alignement path later
        new_item['img_size'] = [image.size[1], image.size[0] ]

        target_width = self.target_width or image.size[0]
        target_height = self.target_height or image.size[1]
        if target_width == -1:
            target_width = int(
                0.5 + 1.0 * target_height / image.height * image.width)
        elif target_height == -1:
            target_height = int(
                0.5 + 1.0 * target_width / image.height * image.height)

        image = image.resize((target_width, target_height),
                             PIL.Image.BICUBIC)

        # From PIL image to Torch tensor [ 32 x w] in this transform
        if self.transform:
            image = self.transform(image)

        # Images will be in W x H x C
        # FIXME: We should not change the layout here, should be handled by the user throught Transforms
        new_item['image'] = image.permute(2, 1, 0)

        # Load the transcription if it exists
        if 'alignment' in new_item:
            if new_item['image'].size(0) != new_item['alignment'].size(0):
                print(f"{new_item['img_filename']} has bad alignment length: "
                      f"image len: {new_item['image'].size(0)} "
                      f"alignment len: {new_item['alignment'].size(0)}")
                raise Exception("Bad alignment length")
        return new_item


def main():
    import torchvision
    import matplotlib.pyplot as plt
    import pprint

    # We should update this test
    dataset = ScribbleLensDataset(root='data/scribblelens.corpus.v1.zip',
                                  dataframe_filename='data/scribblelens.corpus.v1.pkl',
                                  alignment_root='data/scribblelens.paths.1.4.zip',
                                  split='supervised',
                                  colormode='bw')

    item = dataset[0]
    pprint.pprint(item)

    plt.imshow(item['image'][:, :, 0].t(),
               cmap=plt.cm.Greys)
    plt.xticks([])
    plt.yticks([])
    if 'text' in item:
        plt.xlabel(dataset.alphabet.idx2str(item['text'], noBlanks=True))
    plt.show()


if __name__ == '__main__':
    main()
