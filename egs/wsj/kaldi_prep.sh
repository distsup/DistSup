#!/bin/bash
# This script writes data to a new directory suitable for reading with
# the pytorch datasets

set -e
# set -o xtrace
. cmd.sh
. path.sh

# Set the options below to match your dataset
datasets=(train_si284 train_si84 test_dev93 test_eval92 test_eval93)

corpus=wsj
trim=tri4b
ali_cmd=align_fmllr.sh
nj=8

input_dir=.
kaldi_exp_dir=`readlink -e $input_dir`

dir=$DISTSUP_DIR/data/$corpus
echo "Computed with: $0" >> $dir/README

main_train_set=${datasets[0]}

echo "Normalizing with respect to: $main_train_set" >&2
if [ -z $DISTSUP_DIR ]; then
   echo "[ERROR] You have to do source set-env.sh in your distsup directory"
   exit -1
fi

mkdir -p $dir

dir=`readlink -e $dir`

mkdir -p $dir/all

cd $kaldi_exp_dir
stage=0


feat_cmd='compute-fbank-feats --use-energy=true --num-mel-bins=80 '`
    `'IN ark:- | add-deltas ark:- OUT'

if [ $stage -le 0 ]; then
	for dt in ${datasets[*]}
	do
	    cat data/$dt/wav.scp
	done | sort | uniq > $dir/all/wav.scp

	interpolated_feat_cmd=`echo $feat_cmd | \
			sed -e "s;IN;scp:$dir/all/wav.scp;" | \
			sed -e "s;OUT;ark,scp:$dir/all/feats.ark,$dir/all/feats.scp;"`

	echo "fbanks Computed with: $interpolated_feat_cmd" >> $dir/README
	echo "Running $interpolated_feat_cmd"
	eval $interpolated_feat_cmd

	cat $dir/all/feats.scp | sed "s/ [^:]*/ ..\/all\/feats.ark/" > $dir/all/tmp
	mv $dir/all/tmp $dir/all/feats.scp

	for dt in ${datasets[*]}
	do
	    mkdir -p $dir/$dt
	    cp data/$dt/text $dir/$dt/text
	    cp data/$dt/utt2spk $dir/$dt/utt2spk
	    utils/filter_scp.pl $dir/$dt/text $dir/all/feats.scp > $dir/$dt/feats.scp
	done
fi

if [ $stage -le 1 ]; then
    pushd $dir/$main_train_set
    compute-cmvn-stats scp:$dir/$main_train_set/feats.scp $dir/$main_train_set/cmvn
    popd
fi

# MFCCs
feat_cmd='compute-mfcc-feats IN ark:- | add-deltas ark:- OUT'

if [ $stage -le 0 ]; then
	FNAME='mfccs'
	interpolated_feat_cmd=`echo $feat_cmd | \
			sed -e "s;IN;scp:$dir/all/wav.scp;" | \
			sed -e "s;OUT;ark,scp:$dir/all/${FNAME}.ark,$dir/all/${FNAME}.scp;"`

	echo "${FNAME} Computed with: $interpolated_feat_cmd" >> $dir/README
	echo "Running $interpolated_feat_cmd"
	eval $interpolated_feat_cmd

	cat $dir/all/${FNAME}.scp | sed "s/ [^:]*/ ..\/all\/${FNAME}.ark/" > $dir/all/tmp
	mv $dir/all/tmp $dir/all/${FNAME}.scp

	for dt in ${datasets[*]}
	do
	    utils/filter_scp.pl $dir/$dt/text $dir/all/${FNAME}.scp > $dir/$dt/${FNAME}.scp
	done

	pushd $dir/$main_train_set
    	compute-cmvn-stats scp:$dir/$main_train_set/${FNAME}.scp $dir/$main_train_set/${FNAME}_cmvn
    popd
fi

# WAVs
feat_cmd='python $DISTSUP_DIR/distsup/databin/wav_scp_to_mat.py IN OUT'
if [ $stage -le 0 ]; then
	FNAME='wav_as_feats'
	interpolated_feat_cmd=`echo $feat_cmd | \
			sed -e "s;IN;scp:$dir/all/wav.scp;" | \
			sed -e "s;OUT;ark,scp:$dir/all/${FNAME}.ark,$dir/all/${FNAME}.scp;"`

	echo "${FNAME} Computed with: $interpolated_feat_cmd" >> $dir/README
	echo "Running $interpolated_feat_cmd"
	eval $interpolated_feat_cmd

	cat $dir/all/${FNAME}.scp | sed "s/ [^:]*/ ..\/all\/${FNAME}.ark/" > $dir/all/tmp
	mv $dir/all/tmp $dir/all/${FNAME}.scp

	for dt in ${datasets[*]}
	do
	    utils/filter_scp.pl $dir/$dt/text $dir/all/${FNAME}.scp > $dir/$dt/${FNAME}.scp
	done

	pushd $dir/$main_train_set
    	compute-cmvn-stats scp:$dir/$main_train_set/${FNAME}.scp $dir/$main_train_set/${FNAME}_cmvn
    popd
fi

if [ $stage -le 2 ]; then
	echo "Doing the phone alignment"
	for dt in ${datasets[*]}
	do
		./steps/${ali_cmd} --nj $nj data/$dt data/lang exp/$trim ali/$dt
		$DISTSUP_DIR/distsup/databin/get_ctm_prons.sh --use-segments false data/$dt data/lang ali/$dt
	done
fi

if [ $stage -le 3 ]; then
	echo "Doing conversion of alignment"
	for dt in ${datasets[*]}
	do
		$DISTSUP_DIR/distsup/databin/convert_ctmprons.pl ali/$dt/ctm_prons $dir/$dt
	done
fi

if [ $stage -le 4 ]; then
	echo "Extracting vocabs on the main dataset ($dir/$main_train_set)"
	# The character vocabulary
	echo '<eps>' > $dir/$main_train_set/vocabulary.txt
	echo '<unk>' >> $dir/$main_train_set/vocabulary.txt
	echo '<spc>' >> $dir/$main_train_set/vocabulary.txt
	cat $dir/$main_train_set/text \
		| cut -d ' ' -f2- \
		| sed -e 's/ //g' \
		| sed -e 's/\(.\)/\1\n/g' \
		| sort | uniq | grep -v '^$' >> $dir/$main_train_set/vocabulary.txt

	# The phoneme vocabulary
	echo '<eps>' > $dir/$main_train_set/vocabulary_phn.txt
	echo '<unk>' >> $dir/$main_train_set/vocabulary_phn.txt
	cat $dir/$main_train_set/text_phn | cut -d ' ' -f 2- | sed -e 's/ /\n/g' \
		| sort | uniq >> $dir/$main_train_set/vocabulary_phn.txt
fi
