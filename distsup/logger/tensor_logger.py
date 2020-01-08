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

import datetime
import json
import os
import pwd
import socket
import time

import numpy as np

import torch

from tensorboardX import SummaryWriter
import logging

from distsup.configuration import Globals


class TensorLogger(object):
    NO_LOG_STATE, NULL_LOG_STATE, STEP_LOG_STATE = (
            'NO_LOG', 'NULL_LOG', 'STEP_LOG')

    # global cache of sumary writers
    summary_writers = {}
    csv_files = {}

    def __init__(self):
        self.log_state = self.NO_LOG_STATE
        self.summary_writer = None
        self.csv_writer = None
        self.bigquery_writer = None
        self.iteration = None
        self.warned_log_state = False

    def get_summary_writer(self, path):
        if path not in self.summary_writers:
            self.summary_writers[path] = SummaryWriter(path)
        return self.summary_writers[path]

    def get_csv_writer(self, path):
        if path not in self.csv_files:
            fname = '{}/events.{}.{}.csv'.format(
                path, int(time.time()), socket.gethostname())
            self.csv_files[path] = open(fname, 'w')
            self.csv_files[path].write('step,name,value\n')
        return self.csv_files[path]

    def get_bigquery_writer(self, subset):

        if self.bigquery_writer is not None:
            self.bigquery_writer.switch_subset(subset)
            return self.bigquery_writer

        if Globals.remote_log:
            bq_dataset = os.environ['GOOGLE_BIGQUERY_DATASET']
            cred_key = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            return BigQueryWriter(bq_dataset, cred_key=cred_key, subset=subset)
        else:
            return None

    def ensure_no_log(self):
        if self.log_state != self.NO_LOG_STATE:
            raise Exception("Cannot call function during active step")

    def ensure_during_log(self):
        if self.log_state == self.NO_LOG_STATE and not self.warned_log_state:
            logging.warning("Cannot call functions log_XXX() without"
                            " activating step in the logger")
            self.warned_log_state = True

    def is_currently_logging(self):
        self.ensure_during_log()
        return self.log_state == self.STEP_LOG_STATE

    def make_step_log(self, log_dir, iteration):
        self.ensure_no_log()
        self.log_state = self.STEP_LOG_STATE
        self.iteration = iteration
        self.summary_writer = self.get_summary_writer(log_dir)
        self.csv_writer = self.get_csv_writer(log_dir)
        subset = log_dir.rstrip('/').rsplit('/', 1)[-1]
        self.bigquery_writer = self.get_bigquery_writer(subset)

    def make_null_log(self):
        self.ensure_no_log()
        self.log_state = self.NULL_LOG_STATE

    def end_log(self):
        self.ensure_during_log()
        if self.log_state == self.STEP_LOG_STATE:
            for writer in self.summary_writer.all_writers.values():
                writer.flush()
            self.summary_writer = None
            self.csv_writer.flush()
            self.csv_writer = None
            if self.bigquery_writer is not None:
                self.bigquery_writer.flush()
            self.iteration = None
        self.log_state = self.NO_LOG_STATE

    def log_scalar(self, name, value):
        if self.is_currently_logging():
            self.summary_writer.add_scalar(name, value, self.iteration)
            self.csv_writer.write('{},{},{}\n'.format(
                self.iteration, name, value))
            if self.bigquery_writer is not None:
                self.bigquery_writer.write(self.iteration, name, value)

    def log_histogram(self, name, values):
        if self.is_currently_logging():
            self.summary_writer.add_histogram(name, values, self.iteration)

    def log_image(self, tag, img, **kwargs):
        kwargs.setdefault('dataformats', 'HWC')
        if self.is_currently_logging():
            self.summary_writer.add_image(
                tag, img, self.iteration, **kwargs)

    def log_images(self, tag, img, **kwargs):
        kwargs.setdefault('dataformats', 'NHWC')
        C_idx = kwargs['dataformats'].index('C')
        if self.is_currently_logging():
            if img.size(C_idx) == 1:
                factors = [-1] * 4
                factors[C_idx] = 3
                img = img.expand(*factors)
            self.summary_writer.add_images(
                tag, img, self.iteration, **kwargs)

    def log_mpl_figure(self, tag, fig):
        if self.is_currently_logging():
            import matplotlib.backends.backend_agg
            matplotlib.backends.backend_agg.FigureCanvas(fig)
            fig.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8
                                ).reshape(h, w, 3)
            self.summary_writer.add_image(
                tag, img, self.iteration, dataformats='HWC')

    def log_audio(self, tag, audio):
        if self.is_currently_logging():
            self.summary_writer.add_audio(tag, audio, self.iteration)


class BigQueryWriter(object):
    meta_entry_uploaded = False

    def __init__(self, bq_dataset, cred_key, subset='train', upload_every=300):
        from google.cloud import bigquery
        self.client = bigquery.Client()
        self.bq_dataset = bq_dataset
        self._subset = subset
        self.last_upload_attempt = time.time()
        self.upload_every = upload_every
        self._cache = []

    def write(self, iteration, name, val):
        if type(val) is torch.Tensor:
            val = val.item()

        if np.isneginf(val):
            val = "-inf"
        elif np.isinf(val):
            val = "+inf"
        elif np.isnan(val):
            val = "NaN"

        entry = {
            'uuid': Globals.exp_uuid,
            'date_utc': datetime.datetime.utcnow(),
            'subset': self._subset,
            'step': iteration,
            'name': name,
            'value': val,
        }
        self._cache.append(entry)

    def maybe_upload_meta(self):
        if BigQueryWriter.meta_entry_uploaded:
            return

        path = os.path.abspath(Globals.save_dir)
        paths = path.rstrip('/').split('/')
        meta_entry = {
            'uuid': Globals.exp_uuid,
            'date_utc': datetime.datetime.utcnow(),
            'date_local': datetime.datetime.now(),
            'exp_tag': Globals.exp_tag or paths[-2],
            'exp_name': paths[-1],
            'exp_path': path,
            'yaml': Globals.exp_config_fpath,
            'cluster': Globals.cluster,
            'user': pwd.getpwuid(os.getuid()).pw_name,
            'host': socket.gethostname(),
        }
        config_entry = {
            'uuid': Globals.exp_uuid,
            'config': json.dumps(Globals.objects_config),
        }
        # Was it uploaded by some previous experiment?
        q = (f'SELECT uuid FROM {self.bq_dataset}.meta'
             f' WHERE uuid = "{Globals.exp_uuid}" LIMIT 1')
        try:
            if sum(1 for _ in self.client.query(q)) == 0:
                meta_table = self.client.get_table(self.bq_dataset + '.meta')
                cfg_table = self.client.get_table(self.bq_dataset + '.configs')
                self.client.insert_rows(meta_table, [meta_entry])
                self.client.insert_rows(cfg_table, [config_entry])
            BigQueryWriter.meta_entry_uploaded = True
        except Exception as e:
            logging.error(f'BigQuery: {str(e)}')

    def flush(self):
        if time.time() - self.last_upload_attempt < self.upload_every:
            return
        self.last_upload_attempt = time.time()
        self.maybe_upload_meta()
        if len(self._cache) == 0:
            return
        try:
            log_table = self.client.get_table(self.bq_dataset + '.log')
            # BigQuery silently drops extra fields - ensure schema matches
            self.check_schema(log_table, self._cache[0])
            ret = self.client.insert_rows(log_table, self._cache)
            for r in ret:
                if r['errors']:
                    logging.warning(f'BigQuery: {str(r["errors"])}')
            self._cache = []
        except Exception as e:
            logging.error(f'BigQuery: {str(e)}')

    def check_schema(self, table, entry):
        if set(col.name for col in table.schema) != set(entry.keys()):
            logging.warning(f'BigQuery: {table.table_id} schema mismatch')

        dtypes = {
            'DATETIME': [datetime.datetime],
            'STRING': [str],
            'INTEGER': [int],
            'FLOAT': [float, np.float32, np.float64],
        }
        for col in table.schema:
            t = type(entry[col.name])
            if not any(t is tdb for tdb in dtypes[col.field_type]):
                logging.warning(f'BigQuery: Field type mismatch: {col.name}. '
                                f'Got {str(type(entry[col.name]))}, '
                                f'expected {str(dtypes[col.field_type])}.')

    def switch_subset(self, subset):
        self._subset = subset

    def close(self):
        self.flush()
