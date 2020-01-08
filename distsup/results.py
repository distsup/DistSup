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

import glob
import os

import pandas as pd


def read_single_cvs(fpath, exp_id):
    fpath = fpath.replace(os.path.sep * 2, os.path.sep)
    df = pd.read_csv(
        fpath, header=0,
        dtype={'step': 'int32', 'name': 'category', 'value': 'float32'})
    (exp_id, subset, fname) = exp_id.rsplit(os.path.sep, 2)
    df['tstamp'] = int(fname.split('.')[1])
    df['exp_id'] = exp_id
    df['subset'] = subset
    cols = ['exp_id', 'subset', 'step', 'name', 'tstamp']
    df = df.sort_values(cols)
    df = df[cols + ['value']]
    meta = {'exp_path': fpath.rsplit(os.path.sep, 2)[0],
            'exp_id': exp_id}
    return df, meta


def read_csvs(root_dir):
    data_files = glob.glob(f'{root_dir}/**/*.csv', recursive=True)
    common = os.path.commonprefix(data_files)
    data_frames, metas = zip(*[
        read_single_cvs(fpath, fpath[len(common):])
        for fpath in data_files])
    df = pd.concat(data_frames, ignore_index=True)
    df = df.loc[df.groupby(['exp_id', 'subset', 'name', 'step']).tstamp.idxmax()]
    del df['tstamp']
    df = df.reset_index(drop=True)
    meta = pd.DataFrame(metas).drop_duplicates()
    paths = meta.exp_path.str.rsplit(os.path.sep, 2, expand=True)
    meta['exp_tag'] = paths[1]
    meta['exp_name'] = paths[2]
    cols = ['exp_id', 'exp_path', 'exp_tag', 'exp_name']
    meta = meta.sort_values(cols)
    meta = meta[cols]
    meta = meta.reset_index(drop=True)
    df = meta[['exp_id', 'exp_tag', 'exp_name']
              ].merge(df, on='exp_id')
    return df, meta


def _like_clauses(like_patterns):
    where_conds = []
    for field_name, field_like in like_patterns.items():
        if field_like:
            where_conds.append(f'{field_name} LIKE "{field_like}"')
    return where_conds


def _in_clause(name, vals):
    return f"{name} IN ({', '.join(repr(v)for v in vals)})"


def _where_clause(where_conds):
    if where_conds:
        where_clause = "WHERE " + " AND ".join(where_conds)
    else:
        where_clause = ""
    return where_clause


def _bq_query(query):
    # print(query)
    from google.cloud import bigquery
    client = bigquery.Client()
    query_job = client.query(query)
    return query_job.to_dataframe()


def read_bq_experiments(exp_name_like=None, exp_tag_like=None, yaml_like=None,
                        user_like=None,
                        cluster_like=None, host_like=None,
                        exp_ids_in=None
                        ):
    like_clauses = _like_clauses({
        'exp_name': exp_name_like,
        'exp_tag': exp_tag_like,
        'yaml': yaml_like,
        'user': user_like,
        'cluster': cluster_like,
        'host': host_like,
    })
    if exp_ids_in:
        uuid_clauses = [_in_clause('uuid', exp_ids_in)]
    else:
        uuid_clauses = []
    query = f"""
        SELECT *  from results.meta
        {_where_clause(like_clauses + uuid_clauses)}
    """
    df = _bq_query(query)
    df = df.rename(columns={'uuid': 'exp_id'})
    return df


def read_bq_results(exp_ids_or_meta_df, subset_like=None, name_like=None):
    if isinstance(exp_ids_or_meta_df, pd.DataFrame):
        uuids = exp_ids_or_meta_df.exp_id.unique()
        meta_df = exp_ids_or_meta_df
    else:
        uuids = exp_ids_or_meta_df
        meta_df = read_bq_experiments(exp_ids_in=uuids)

    uuid_clauses = [_in_clause('uuid', uuids)]
    like_clauses = _like_clauses({
        'subset': subset_like,
        'name': name_like
    })
    log_df = _bq_query(f"""
    select * from results.log
    {_where_clause(like_clauses + uuid_clauses)}
    """)
    log_df = log_df.rename(columns={'uuid': 'exp_id'})
    log_df = log_df.loc[
        log_df.groupby(['exp_id', 'subset', 'name', 'step']).date_utc.idxmax()]
    log_df = meta_df[['exp_id', 'exp_tag', 'exp_name']
                     ].merge(log_df, on='exp_id')
    log_df = log_df.reset_index(drop=True)
    return log_df, meta_df
