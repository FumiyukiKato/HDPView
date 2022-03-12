import numpy as np

# Not p-view-style data
# perurbed count tensor format
# e.g., Identity and DAWA
class NoisedData:
    def __init__(self, columns, data, workload_optimized=False):
        self.columns = columns
        self.data = data
        self.workload_optimized = workload_optimized
    
    def project(self, cols):
        if self.workload_optimized:
            if type(cols) is not tuple:
                cols = (cols,)
            return self.data[cols].ravel()
        else:
            if type(cols) is not tuple:
                cols = (cols,)
            aggregate_cols = filter(lambda x: x not in cols, self.columns)
            indices = tuple([self.columns.index(col) for col in aggregate_cols])
            return self.data.sum(axis=indices).ravel()
        

class SynData:
    """From synthetic data for counting task
    """
    def __init__(self, syn_df, cardinality_dict):
        self.cardinality_dict = cardinality_dict
        self.syn_df = syn_df
        
    def project(self, proj):
        count_table = self.syn_df[list(proj)].groupby(list(proj)).size()
        cardinality_list = [ self.cardinality_dict[column] for column in proj ]
        X = np.zeros(cardinality_list)
        for index, count in count_table.iteritems():
            X[index] = count
        return X.ravel()

    def run_query(self, query):
        boolean_index = (self.syn_df[query.conditions[0].attribute] >= query.conditions[0].start) & (self.syn_df[query.conditions[0].attribute] <= query.conditions[0].end)
        for cond in query.conditions[1:]:
            boolean_index = boolean_index & (self.syn_df[cond.attribute] >= cond.start) & (self.syn_df[cond.attribute] <= cond.end )
        return boolean_index.sum()