#!/bin/bash

if [ "$DISTSUP_DIR" == "" ]; then
    echo "Source set-env.sh before running this script!"
    exit 1
fi

if [ ! -f $DISTSUP_DIR/data/scribblelens.corpus.v1.2.zip ]; then
    echo "Downloading scribblelens corpus..."
    mkdir -p $DISTSUP_DIR/data
    url=http://www.openslr.org/resources/84/scribblelens.corpus.v1.2.zip
    wget $url -O $DISTSUP_DIR/data/scribblelens.corpus.v1.2.zip || exit 1
    echo "Done"
fi

if [ ! -f $DISTSUP_DIR/data/scribblelens.paths.1.4b.zip ]; then
    echo "Extracting alignments..."
    7z e $DISTSUP_DIR/data/scribblelens.corpus.v1.2.zip \
        -o$DISTSUP_DIR/data/ \
        scribblelens.corpus.v1/corpora/scribblelens.paths.1.4b.zip
    echo "Done"
fi


echo "Everything downloaded"
