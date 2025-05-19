import pandas as pd
import numpy as np
from utils import *
import time

class CRISMER:
    def __init__(self, model, bins, weights,opti_th=None, ref_genome=None, scaler=None):
        self.model = model
        self.bins = bins
        self.prob_weight = weights
        self.ref_genome = ref_genome
        if opti_th is not None:
            self.opti_th = opti_th
        else:
            self.opti_th = 0.77
        self.proj_t = time.time()
        self.scaler = scaler

    def score(self, df):
        df.reset_index(inplace=True)
        if 'On' not in df.columns or 'Off' not in df.columns:
            scored_df = pd.DataFrame({
                'On': df[0],
                'Off': df[3]
            })
        else:
            scored_df = df
        
        test_x = one_hot_features(scored_df)
        return get_score_without_label(self.model, test_x,scaler=self.scaler,T=10)
        

    def single_score_(self, new_sgrna, target):
        df = pd.DataFrame({'On': [new_sgrna], 'Off': [target]})
        test_x = one_hot_features(df)
        return get_score_without_label(self.model, test_x, scaler=self.scaler,T=10) 

    def score_bin_(self, y_pred):
        if y_pred.shape[0] != 0:
            y_df = pd.DataFrame(y_pred.reshape(-1, 1), columns=['CRISMER-Score'])
            return y_df['CRISMER-Score'].value_counts(bins=self.bins, sort=False).values
        return np.array([0] * (len(self.bins) - 1))

    def single_aggre_(self, y_pred, out_cnt=True):
        cnt = self.score_bin_(y_pred)
        aggre = (cnt * self.prob_weight).sum()
        return np.append(cnt[-4:], aggre) if out_cnt else aggre

    def single_spec_(self, y_pred):
        aggre = self.single_aggre_(y_pred, out_cnt=False)
        return 200 / (200 + aggre)

    def spec_per_sgRNA(self, data_path=None, data_df=None, On='On', Off='Off', target=None, out_df=False):
        if data_df is not None:
            data_set = data_df
        else:
            data_set = pd.read_csv(data_path, sep=",", header=0, index_col=None)
        
        offt = data_set.loc[:, Off].values
        if target is not None:
            assert len(target) == 23, 'target sequence must have 23 nt'
            assert data_set.loc[offt == target, :].shape[0] > 0, 'No sequence match the target sequence'
            data_set = pd.concat([data_set.loc[offt == target, :], data_set.loc[offt != target, :]])
        
        y_pred = self.score(data_set)
        spec = self.single_spec_(y_pred[1:])
        
        if out_df:
            data_set['CRISMER-Score'] = y_pred
            return spec, data_set
        else:
            return spec

    def CasoffinderSpec_(self, sgrna, target, out_df=False, offtar_search=None, mm=6, dev='G0'):
        if offtar_search is None:
            offtar_search = os.path.join(pwd, 'script/casoffinder_genome.sh')
        if os.path.exists(f'.temp_{self.proj_t}_casoffinder.out'):
            os.remove(f'.temp_{self.proj_t}_casoffinder.out')
        if os.path.exists(f'.temp_{self.proj_t}_casoffinder.in'):
            os.remove(f'.temp_{self.proj_t}_casoffinder.in')
        os.system("{} {} {} {} {} {}".format(offtar_search, sgrna[:20], self.ref_genome, mm, dev, self.proj_t))
        data_set = pd.read_csv(f'.temp_{self.proj_t}_casoffinder.out', sep="\t", header=None, index_col=None)
        offt = data_set.loc[:, 3].values
        offt = np.array([str.upper(t) for t in offt])
        data_set.loc[:, 3] = offt
        data_set = data_set[-data_set[3].str.contains('N|R|W|M|V|Y|K|D|S|J')]
        data_set.drop_duplicates([1, 2, 3], inplace=True)
        if data_set[data_set[3].str.contains(target[:20])].shape[0] == 0:
            data_set = pd.concat([data_set, pd.DataFrame([[data_set.iloc[0, 0], np.nan, np.nan, target, np.nan, np.nan]])], ignore_index=True)
        data_set = pd.concat(
            [data_set[data_set[3].str.contains(target[:20])], data_set[-data_set[3].str.contains(target[:20])]])
        if out_df:
            spec, out_dset = self.spec_per_sgRNA(data_df=data_set, On=0, Off=3, out_df=out_df)
        else:
            spec = self.spec_per_sgRNA(data_df=data_set, On=0, Off=3)
        os.remove(f".temp_{self.proj_t}_casoffinder.out")
        
        if out_df:
            return spec, out_dset
        else:
            return spec

    def opti(self, target, opti_type=None, ref=None, offtar_search=None, mm=6, dev='G0'):
        # opti_type: how to mutate the sgRNA? default is 3 types of mutations for each position
        if opti_type == None:
            opti_pos = []
            opti_seq = []
            for p in range(1, 21):
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif type(opti_type) == tuple:
            opti_pos = []
            opti_seq = []
            for p in range(opti_type[0], opti_type[1]):
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif (type(opti_type) == np.ndarray) or (type(opti_type) == list) or (type(opti_type) == range):
            opti_pos = []
            opti_seq = []
            for p in opti_type:
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif type(opti_type) == pd.core.frame.DataFrame:
            opti_nt = opti_type
        else:
            opti_nt = pd.read_csv(opti_type, header=0, index_col=0)


        results = []
        # calculate WT spec score
        spec, out_df = self.CasoffinderSpec_(target, target, out_df=True, offtar_search=offtar_search, mm=mm, dev=dev)
        
        cnt_results = self.score_bin_(out_df.loc[:, 'CRISMER-Score'].values[1:])[-8:]
        cnt_results = np.append(cnt_results[:3], cnt_results[3:].sum())
        if ref is not None:
            ref_merge = ref.loc[:, ['Offtarget_Sequence', 'GUIDE-Seq Reads']]
            ref_merge.columns = [3, 'Reads']
            out_merge = pd.merge(out_df.iloc[1:,:], ref_merge, how='left', on=[3], sort=False)
            out_merge.fillna(0, inplace=True)
            out_merge = out_merge.loc[out_merge.loc[:,'Reads'] > 0, :]
            if out_merge.shape[0] != 0:
                out_merge.to_csv('{}_ref_match.tsv'.format(target), sep='\t')
                spec_ref = self.single_spec_(out_merge.loc[:, 'CRISMER-Score'].values[1:], out_cnt=False)
            else:
                spec_ref = 1.0
            results.append([target, target, 'WT', out_df.loc[:, 'CRISMER-Score'].values[0]] + list(cnt_results) + [spec, spec_ref])
        else:
            results.append([target, target, 'WT', out_df.loc[:, 'CRISMER-Score'].values[0]] + list(cnt_results) + [spec])
        # calculate spec scores of modified sgRNAs
        for i in range(opti_nt.shape[0]):
            pos = int(opti_nt.loc[i, 'Pos'])
            nt = opti_nt.loc[i, 'nt']
            if target[pos - 1] != nt:
                new_sgrna = target[:pos - 1] + nt + target[pos:]
                if self.single_score_(new_sgrna, target) > 0.5:
                
                    spec, out_df = self.CasoffinderSpec_(new_sgrna, target, out_df=True, offtar_search=offtar_search, mm=mm, dev=dev)
                    cnt_results = self.score_bin_(out_df.loc[:, 'CRISMER-Score'].values[1:])[-5:]

                    cnt_results = np.append(cnt_results[:3], cnt_results[3:].sum())
                    if ref is not None:
                        out_merge = pd.merge(out_df.iloc[1:,:], ref_merge, how='left', on=[3], sort=False)
                        out_merge.fillna(0, inplace=True)
                        out_merge = out_merge.loc[out_merge.loc[:,'Reads'] > 0, :]
                        if out_merge.shape[0] != 0:
                            spec_ref = self.single_spec_(out_merge.loc[:, 'CRISMER-Score'].values[1:], out_cnt=False)
                        else:
                            spec_ref = 1.0
                    if ref is not None:
                        results.append([new_sgrna, target, '{}{}>{}'.format(target[pos - 1], pos, nt), out_df.loc[:, 'CRISMER-Score'].values[0]] + list(cnt_results) + [spec, spec_ref])
                    else:
                        results.append([new_sgrna, target, '{}{}>{}'.format(target[pos - 1], pos, nt), out_df.loc[:, 'CRISMER-Score'].values[0]] + list(cnt_results) + [spec])

        results = np.array(results)
        if results.shape[0] == 1:
            results = np.vstack([results, np.array([[np.nan for i in range(results.shape[1])]], dtype=object)])
            results[1, 1:3] = np.array([target, 'Optimization unavailable'])
        
        if ref is not None:
            print(results.head())
            results_df = pd.DataFrame(results, columns=['sgRNA', 'Target', 'Mutation', 'CRISMER-Score', '(0.65, 0.7]', '(0.7, 0.75]', 
                                                        '(0.75, 0.8]', '(0.8, 1.0]', 'CRISMER-Spec', 'ref_Spec'])
        else:
            results_df = pd.DataFrame(results, columns=['sgRNA', 'Target', 'Mutation', 'CRISMER-Score', '(0.65, 0.7]', '(0.7, 0.75]', 
                                                        '(0.75, 0.8]', '(0.8, 1.0]', 'CRISMER-Spec'])
        results_df.iloc[:, 3:] = np.array(results_df.iloc[:, 3:].values, dtype='float')
        try:
            results_df.iloc[:, 4:8] = np.array(results_df.iloc[:, 4:8].values, dtype='int')
        except:
            results_df.iloc[0, 4:8] = np.array(results_df.iloc[0, 4:8].values, dtype='int')
        results_df['delta_Spec'] = results_df.loc[:, 'CRISMER-Spec'].values - results_df.loc[0, 'CRISMER-Spec']
        results_df['rk'] = np.array([0] + [1] * (results_df.shape[0]-1))

        results_out = results_df.sort_values(by=['rk', 'delta_Spec'], ascending=[True, False])
        
        results_out.index = np.arange(results_out.shape[0])

        
        return results_out.iloc[:, :-1]

    