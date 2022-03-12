import numpy as np
import pandas as pd
from query import QueryCondition

"""
CountTable is data model in DP-Mondrian cutting.
It implicitly has count tensor and abstracts the cutting process for count tensor.

NoisedCountTable is representation of p-view in DP-Mondrian.
"""

EMPTY_DOMAIN = (0, -1)
        
class CountTable:
    """CountTable
        Efficient count data representation compared to count tensor.

    Attributes:
        pd_table (pd.DataFrame): count data table as pandas table
        cardinality_dict ([int]): cardinalities of original data schema
        index ([int]): indices
        domain_dict ({int: (int, int)}): range for columns
        original_shape ((int)): cardinalities
    """
    def __init__(self, pd_table, cardinality_dict, index, domain_dict, depth=0, id=0, seed=0):
        self.id = id
        self.seed = seed
        self.prng = np.random.RandomState((id+seed) % 4294967295)
        self.pd_table = pd_table
        self.cardinality_dict = cardinality_dict
        self.index = index
        self.domain_dict = domain_dict
        self.original_shape = tuple(cardinality_dict.values())
        self.depth = depth
    
    def set_random(self, seed):
        self.seed = seed
        self.prng = np.random.RandomState(self.id+seed)

    def project(self, attrs):
        return CountTable(
            pd_table=self.pd_table.groupby(list(attrs)).sum(),
            cardinality_dict={ attr: self.cardinality_dict[attr] for attr in attrs },
            index=attrs,
            domain_dict={ attr: self.domain_dict[attr] for attr in attrs }
        )
        
    def info(self):
        print(self.cardinality_dict)
        print("total domain: ", self.size())

    def split_values(self, dim, ith):
        cardinality_dict = self.cardinality_dict.copy()
        cardinality = cardinality_dict[dim]

        left_boolean = self.pd_table.index.get_level_values(dim) <= self.domain_dict[dim][0] + ith
        right_boolean = np.logical_not(left_boolean)

        cardinality_dict[dim] = ith + 1
        left_zero_num = np.prod(list(cardinality_dict.values()), dtype=np.float)  - left_boolean.sum()
        left_value = implicit_ae(self.pd_table.loc[left_boolean].values, left_zero_num)

        cardinality_dict[dim] = cardinality - (ith + 1)
        right_zero_num = np.prod(list(cardinality_dict.values()), dtype=np.float)  - right_boolean.sum()
        right_value = implicit_ae(self.pd_table.loc[right_boolean].values, right_zero_num)

        return left_value, right_value
    
    def range_query(self, query):
        """Query processing for count table.

        Args:
            query (Query): query

        Returns:
            np.array: 
            int: number of zero counts

        Examples:
            >>> from TableGenerator import sample5
            >>> from Query import Query, QueryCondition
            >>> table = pd.DataFrame(sample5)
            >>> ct = CountTable.from_pd_table(table, [10,10,10], table.columns)
            >>> ct.range_query(Query([QueryCondition(0, 0, 2), QueryCondition(1, 0, 2)]))
            (array([[13],
                   [ 3],
                   [ 4],
                   [ 3]]), 86)
        
        """
        conditions = query.conditions
        boolean_index = True
        domain_dict = self.domain_dict.copy()
        for condition in conditions:
            assert self.domain_dict.get(condition.attribute), "Error: No attribute"
            boolean_index = boolean_index & (self.domain_dict[condition.attribute][0] + condition.start <= self.pd_table.index.get_level_values(condition.attribute)) & (self.pd_table.index.get_level_values(condition.attribute) <= self.domain_dict[condition.attribute][0] + condition.end)
            self.update_domain(condition, domain_dict)
        return self.pd_table.loc[boolean_index].values, self.count_domain_dict(domain_dict) - boolean_index.sum()
    
    def update_domain(self, condition, domain_dict):
        """Update block's domain ranges with range condition.

        """
        if domain_dict[condition.attribute][0] <= condition.start and condition.start <= domain_dict[condition.attribute][1]:
            domain_dict[condition.attribute] = (condition.start, domain_dict[condition.attribute][1])
        if domain_dict[condition.attribute][0] <= condition.end and condition.end <= domain_dict[condition.attribute][1]:
            domain_dict[condition.attribute] = (domain_dict[condition.attribute][0], condition.end)
        if (condition.end < domain_dict[condition.attribute][0]) or (domain_dict[condition.attribute][1] < condition.start):
            domain_dict[condition.attribute] = EMPTY_DOMAIN
        
    def get_sub_block_values(self, block_ranges):
        """
        Args:
            block_ranges: {dimension: (int)}
        """
        boolean_index = True
        domain_dict = self.domain_dict.copy()
        for dim, block_range in block_ranges:
            assert self.domain_dict.get(dim), "Error: No attribute"            
            boolean_index = boolean_index & (self.domain_dict[dim][0] + block_range[0] <= self.pd_table.index.get_level_values(dim)) & (self.pd_table.index.get_level_values(dim) <= self.domain_dict[dim][0] + block_range[1])
            condition = QueryCondition(dim, block_range[0], block_range[1])
            self.update_domain(condition, domain_dict)
        return self.pd_table.loc[boolean_index].values, self.count_domain_dict(domain_dict) - boolean_index.sum()

    def split(self, d, i):
        """Split block into 2 blocks at `d` th dimension `i` th bin.

        Args:
            d (int): dimension number
            i (int): domain number

        Returns:
            CountTable: left block
            CountTable: right block

        Examples:
            >>> from TableGenerator import sample5
            >>> from Query import Query, QueryCondition
            >>> table = pd.DataFrame(sample5)
            >>> ct = CountTable.from_pd_table(table, [10,10,10], table.columns)
            >>> left, right = ct.split(0, 2)
        """
        boolean_index = (self.pd_table.index.get_level_values(d) <= self.domain_dict[d][0]+ i)
        
        left_table = self.pd_table.loc[boolean_index]
        left_cardinality_dict = self.cardinality_dict.copy()
        left_cardinality_dict[d] = i + 1
        left_domain_dict = self.domain_dict.copy()
        left_domain_dict[d] = (self.domain_dict[d][0], self.domain_dict[d][0] + i )
        left_block = CountTable(left_table, left_cardinality_dict, self.index, left_domain_dict, self.depth + 1, id=2*self.id+1, seed=self.seed)
        
        right_table = self.pd_table.loc[np.logical_not(boolean_index)]
        right_cardinality_dict = self.cardinality_dict.copy()
        right_cardinality_dict[d] = self.cardinality_dict[d] - (i + 1)
        right_domain_dict = self.domain_dict.copy()
        right_domain_dict[d] = (self.domain_dict[d][0] + i + 1, self.domain_dict[d][1])
        right_block = CountTable(right_table, right_cardinality_dict, self.index, right_domain_dict, self.depth + 1, id=2*self.id+2, seed=self.seed)
        return left_block, right_block
                        
    def count_domain_dict(self, domain_dict):
        """All counts from domain dict
        """
        cardinality_list = []
        for dim, val in domain_dict.items():
            cardinality_list.append(val[1] - val[0] + 1)
        return np.prod(cardinality_list, dtype=np.float)
    
    def values(self):
        """Non-zero values from table
        """
        return self.pd_table.values
     
    def zero_num(self):
        """The number of zero values.
        """
        return np.prod(list(self.cardinality_dict.values()), dtype=np.float) - len(self.pd_table.values)
    
    def size(self):
        """Total size of count tensor.
        """
        return np.prod(list(self.cardinality_dict.values()), dtype=np.float)
    
    def total_count(self):
        """Total value of record counts.
        """
        return self.pd_table.values.sum()

    def run_query(self, query):
        """Run range count query.

        Examples:
            >>> from TableGenerator import sample5
            >>> from Query import Query, QueryCondition
            >>> table = pd.DataFrame(sample5)
            >>> ct = CountTable.from_pd_table(table, [10,10,10], table.columns)
            >>> q = Query([QueryCondition(0, 0, 3)])
            >>> ct.run_query(q)
            29
            >>> q = Query([QueryCondition(0, 0, 3), QueryCondition(1, 0, 1)])
            >>> ct.run_query(q)
            23
        """
        values, _ = self.range_query(query)
        return np.sum(values)
    
    @classmethod
    def from_dataset(cls, dataset):
        """Construct from pandas table.
        Args:
            dataset (Dataset)
        """
        domain_dict = {}
        cardinality_dict = {}
        for column, cardinality in dataset.domain.config.items():
            domain_dict[column] = (0, cardinality-1)
        for column, cardinality in dataset.domain.config.items():
            cardinality_dict[column] = cardinality
        count_table = pd.DataFrame(dataset.df.groupby(list(dataset.df.columns)).size())
        count_table.columns = ["count"]
        
        return CountTable(count_table, cardinality_dict, dataset.domain.attrs, domain_dict)
    
    @classmethod
    def from_pd_table(cls, table, cardinality_list, columns):
        """Construct from pandas table.
        Args:
            table (pd.DataFrame): raw record table
            cardinality_list ([int]): cardinalities of original data schema
            columns ([column]):  columns of raw record table
        Examples:
            >>> from TableGenerator import sample5
            >>> from Query import Query, QueryCondition
            >>> table = pd.DataFrame(sample5)
            >>> ct = CountTable.from_pd_table(table, [10,10,10], table.columns)
        """
        domain_dict = {}
        cardinality_dict = {}
        for column, cardinality in zip(columns, cardinality_list):
            domain_dict[column] = (0, cardinality-1)
        for column, cardinality in zip(columns, cardinality_list):
            cardinality_dict[column] = cardinality
        count_table = pd.DataFrame(table.groupby(list(table.columns)).size())
        count_table.columns = ["count"]
        
        return CountTable(count_table, cardinality_dict, columns, domain_dict)
    
    @classmethod
    def from_pd_count_table(cls, count_table, cardinality_list, columns):
        """Construct from pandas count table.
        Args:
            count_table (pd.DataFrame): table including count values
            cardinality_list ([int]): cardinalities of original data schema
            columns ([column]):  columns of raw record table
        """
        domain_dict = {}
        cardinality_dict = {}
        for column, cardinality in zip(columns, cardinality_list):
            domain_dict[column] = (0, cardinality-1)
        for column, cardinality in zip(columns, cardinality_list):
            cardinality_dict[column] = cardinality
        count_table.columns = ["count"]
        
        return CountTable(count_table, cardinality_dict, columns, domain_dict)
    
class Block:
    """Block

    Attributes:
        ranges ({int: (int, int)}): ranges as tuple for each dimension
        value (float): aggregated and averaged values for each counts
    """
    def __init__(self, ranges, value):
        self.ranges = ranges
        size_list = []
        for d, r in ranges.items():
            size_list.append(r[1] - r[0] + 1)
        self.size = np.prod(size_list, dtype=np.float)
        self.value = value

    def random_sample(self, prng):
        """Sample one record from this block.

        Attributes:
            prng (np.random.RandomState): random state
        """
        return { k: prng.choice(np.arange(r[0], r[1]+1)) for k, r in self.ranges.items() }

class NoisedCountTable:
    """NoisedCountTable
        Efficient count data representation compared to count tensor.
        Noised count data.

    Attributes:
        cardinality_dict ([int]): cardinalities of original data schema
        index ([int]): indices
        domain_dict ({int: (int, int)}): range for columns
        blocks ([Block]): list of blocks
        original_shape ((int)): cardinalities
    """
    def __init__(self, cardinality_dict, index, domain_dict, block_result_list):
        self.cardinality_dict = cardinality_dict
        self.index = index
        self.domain_dict = domain_dict
        self.blocks = [ 
            Block(
                block_result.domain_dict,
                self.perturb(block_result.domain_dict, block_result.perturbation_error, block_result.mean)
            ) for block_result in block_result_list 
        ]
        self.original_shape = tuple(cardinality_dict.values())
        
    def project(self, attrs):
        """Materialized data vector to answer ektelo.workload.
        Args:
            attrs (tuple(str)): ex ("race", "sex")
        Returns:
            1-d numpy array
        """
        if type(attrs) is not tuple:
            attrs = (attrs,)
        if type(attrs) is list:
            attrs = tuple(attrs)
        vector_shape = [self.cardinality_dict[attr] for attr in attrs ]
        datavector = np.zeros(vector_shape)
        for block in self.blocks:
            indices = [ range(block.ranges[attr][0], block.ranges[attr][1]+1) for attr in attrs ]
            size = np.prod([ block.ranges[attr][1] - block.ranges[attr][0] + 1 for attr in attrs ], dtype=np.float)
            datavector[np.ix_(*indices)] += block.size*block.value / size
        return datavector.flatten()

    def range_count_query(self, query):
        """Query processing for count table.

        Args:
            query (Query): query

        Returns:
            float: estimated count
        """
        domain_dict = self.domain_dict.copy()
        for condition in query.conditions:
            self.update_domain(condition, domain_dict)
        return np.sum([ self.count_in_block(domain_dict, block) for block in self.blocks ])
    
    def update_domain(self, condition, domain_dict):
        if domain_dict[condition.attribute][0] <= condition.start and condition.start <= domain_dict[condition.attribute][1]:
            domain_dict[condition.attribute] = (condition.start, domain_dict[condition.attribute][1])
        if domain_dict[condition.attribute][0] <= condition.end and condition.end <= domain_dict[condition.attribute][1]:
            domain_dict[condition.attribute] = (domain_dict[condition.attribute][0], condition.end)
        if (condition.end < domain_dict[condition.attribute][0]) or (domain_dict[condition.attribute][1] < condition.start):
            domain_dict[condition.attribute] = EMPTY_DOMAIN  

    def count_in_block(self, domain_dict, block):
        block_domain_dict = domain_dict.copy()
        for dim, single_range in block.ranges.items():
            if block_domain_dict[dim][0] <= single_range[0] and single_range[0] <= block_domain_dict[dim][1]:
                block_domain_dict[dim] = (single_range[0], block_domain_dict[dim][1])
            if block_domain_dict[dim][0] <= single_range[1] and single_range[1] <= block_domain_dict[dim][1]:
                block_domain_dict[dim] = (block_domain_dict[dim][0], single_range[1])
            if (single_range[1] < block_domain_dict[dim][0]) or (block_domain_dict[dim][1] < single_range[0]):
                # 範囲になかったら，即終了する
                return 0
        return self.count_domain_dict(block_domain_dict)*block.value
    

    def count_domain_dict(self, domain_dict):
        """All counts from domain dict
        """
        cardinality_list = []
        for dim, val in domain_dict.items():
            cardinality_list.append(val[1] - val[0] + 1)
        return np.prod(cardinality_list, dtype=np.float)
    
    def run_query(self, query):
        return self.range_count_query(query)
    
    def perturb(self, ranges, noise, mean):
        block_size = self.count_domain_dict(ranges)
        return mean + (noise / block_size)

    def random_sample(self, prng, sample_size):
        """Random sampling from noised histgram.
            
        Args:
            prng (np.random.RandomState): random state
            sample_size (int): sample size

        Returns:
            [[int]]: list of record points
        """
        p = normalize(np.array([ block.value * block.size for block in self.blocks ]))
        sampled_blocks = prng.choice(self.blocks, sample_size, p=p, replace=True)

        sampled = [ block.random_sample(prng) for block in sampled_blocks ]
        
        return sampled

    @classmethod                                   
    def from_count_table(cls, count_table, block_result_list):
        return NoisedCountTable(count_table.cardinality_dict, count_table.index, count_table.domain_dict, block_result_list)

def work(params):
    prng, block = params
    return block.random_sample(prng)

def normalize(x, axis=None):
    """Normalization.
        
    Args:
        x (np.array): data
        axis (int): axis

    Returns:
        np.array: normalized data
    
    Examples:
        >>> data = np.array([1.5, 0.3, -0.8])
        >>> normalize(data)
        array([0.67647059, 0.32352941, 0.        ])
    """    
    min = x.min(axis=axis, keepdims=True)
    result = x-min
    result = result/result.sum()
    return result


def implicit_ae(data, zero_num):
    """Calculate AE from sparse block format
    """
    mean = np.sum(data) / (len(data) + zero_num)
    return np.sum(np.absolute(data - mean)) + np.absolute(mean)*zero_num


class BlockResult:
    def __init__(self, domain_dict, depth, mean=0, aggregation_error=0, perturbation_error=0):
        self.domain_dict = domain_dict
        self.depth = depth
        self.mean = mean
        self.aggregation_error = aggregation_error
        self.perturbation_error = perturbation_error

    def __str__(self):
        return f'range: {self.domain_dict}]\ndepth: {self.depth}\nmean: {self.mean}\naggregation_error: {self.aggregation_error}\nperturbation_error: {self.perturbation_error}\n'


def calculate_mean_and_aggregation_error(block, ranges):
    """Calculate mean and aggregation error for given blocks as block and ranges.

    Args:
        block (CountTable): count block
        ranges ([{int, (int, int)}]): [{dimension, (start, end)}]
    """
    values, zero_num = block.get_sub_block_values(list(ranges.items()))
    assert len(values) + zero_num > 0, "Error: range is invalid"
    mean = values.sum() / (len(values) + zero_num)
    ae = implicit_ae(values, zero_num)
    return mean, ae