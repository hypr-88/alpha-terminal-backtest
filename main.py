import pandas as pd
import numpy as np
from scipy.stats import rankdata
import requests
import talib
from arch import arch_model
#np.random.seed(2021)
'''
Notes: s stands for float, v stands for np.ndarray, m stands for np.ndarray
Definition of 67 functions of operations:
    1. s = s + s
    2. s = s - s
    3. s = s * s
    4. s = s / s
    5. s = |s|
    6. s = 1 / s
    7. s = sin(s)
    8. s = cos(s)
    9. s = tan(s)
    10. s = arcsin(s)
    11. s = arccos(s)
    12. s = arctan(s)
    13. s = e^s
    14. s = log(s)
    15. s = 1, if s>0
            0, otherwise
    16. v[i] = 1 if v[i] > 0
                0, otherwise
    17. m[i,j] = 1 if m[i,j] > 0
                0, otherwise
    18. v[i] = s * v[i]
    19. v[i] = s, for all i
    20. v[i] = 1 / v[i]
    21. s = norm(v)
    22. v[i] = |v[i]|
    23. v[i] = v[i] + v[i]
    24. v[i] = v[i] - v[i]
    25. v[i] = v[i] * v[i]
    26. v[i] = v[i] / v[i]
    27. s = dot(v, v)
    28. m[i,j] = v[i]*v[j]
    29. m[i,j] = s * m[i,j]
    30. m[i,j] = 1 / m[i,j]
    31. v = dot(m, v)
    32. m[,j] = v[j]
    33. m[i,] = v[i]
    34. s = norm(m)
    35. v[i] = norm(m[i,])
    36. v[j] = norm(m[,j])
    37. m = transpose(m)
    38. m[i,j] = |m[i,j]|
    39. m[i,j] = m[i,j] + m[i,j]
    40. m[i,j] = m[i,j] - m[i,j]
    41. m[i,j] = m[i,j] * m[i,j]
    42. m[i,j] = m[i,j] / m[i,j]
    43. m = matmul(m, m)
    44. s = min(s1, s2)
    45. v[i] = min(v1[i], v2[i])
    46. m[i,j] = min(m1[i,j], m2[i,j])
    47. s = max(s1, s2)
    48. v[i] = max(v1[i], v2[i])
    49. m[i,j] = max(m1[i,j], m2[i,j])
    50. s = mean(v)
    51. s = mean(m)
    52. v[i] = mean(m[i,])
    53. v[i] = std(m[i,])
    54. s = std(v)
    55. s = std(m)
    56. s = const
    57. v[i] = const, for all i
    58. m[i,j] = const, for all i,j
    59. s ~ Uniform(a, b)
    60. v[i] ~ Uniform(a, b)
    61. m[i,j] ~ Uniform(a, b)
    62. s ~ Normal(a, b)
    63. v[i] ~ Normal(a, b)
    64. m[i,j] ~ Normal(a, b)
    65. s = rank(s)
    66. s = rankIndustry(s)
    67. s = s - mean(s_in_same_Industry)
    
'''
class Alpha():
    def __init__(self):
        self.symbolList = []
        self.windowSize = 0
        self.nodes = []
        self.setupOPs = []
        self.predictOPs = []
        self.updateOPs = []
        self.operandsValues = {}
        
        self.extractInput()
        self.simplifyOperations()
        
    def extractInput(self):
        '''request API here and update attribute'''
        url = 'http://13.113.253.201/api/alpha'
        
        payload = {'name': name}
        
        headers = {'Token': 'q0hcdABLUhGAzW3j'}
        
        response = requests.get(url = url, headers = headers, params = payload).json()
        
        if response['status']:
            response = response['data']
            self.symbolList = response['symbolList'] 
            self.windowSize = response['window']
            self.nodes = response['nodes']
            self.setupOPs = response['setupOPs'] 
            self.predictOPs = response['predictOPs'] 
            self.updateOPs = response['updateOPs'] 
            self.operandsValues = response['operandsValues']
    def simplifyOperations(self):
        self.operations = self.setupOPs + self.predictOPs
        
#def init():
alpha = Alpha()
windowSize = alpha.windowSize
operations = alpha.operations
operandsValues = alpha.operandsValues
symbolList = alpha.symbolList
featuresList = ['open', 'high', 'low', 'close', 'volume', 'VWAP', 'open_return', 'high_return', 'low_return', 'close_return', 
                'log volume', 'log volatility', 'open-close', 'log high-low', 'EMA5', 'EMA10', 'EMA20', 'EMA30', 'STD5', 'STD10', 'STD20', 'STD30', 
                'BB high', 'BB mid', 'BB low', 'MACD fast', 'MACD slow', 'MACD'] 
initialBalance = exchange.GetAccount()['Balance']/len(symbolList)
initialStocks = [ex.GetAccount()['Stocks'] for ex in exchanges]

def getData(ex):
    df = pd.DataFrame(ex.GetRecords(PERIOD_M1))
    return df

def processData(raw_df):
    raw_df.drop('OpenInterest', axis=1, inplace=True)
    raw_df.rename(
        columns={'Time': 'time', 'Open': 'open', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'},
        inplace=True)
    raw_df['time'] = pd.to_datetime(raw_df['time'])
    df = pd.DataFrame()
    df['open'] = raw_df.groupby(pd.Grouper(key = 'time', freq = '10min'))['open'].first()
    df['high'] = raw_df.groupby(pd.Grouper(key = 'time', freq = '10min'))['high'].max()
    df['low'] = raw_df.groupby(pd.Grouper(key = 'time', freq = '10min'))['low'].min()
    df['close'] = raw_df.groupby(pd.Grouper(key = 'time', freq = '10min'))['close'].last()
    df['volume'] = raw_df.groupby(pd.Grouper(key = 'time', freq = '10min'))['volume'].sum()
                    
    raw_df['vol x close'] = df['close']*df['volume']
    df['VWAP'] = raw_df.groupby(pd.Grouper(key = 'time', freq = '10min'))['vol x close'].sum()/df['volume']
    df.dropna(inplace = True)
    if len(df.index) < 2*windowSize: return None
    df['open_return'] = df['open']/df['open'].shift(1) - 1
    df['high_return'] = df['high']/df['high'].shift(1) - 1
    df['low_return'] = df['low']/df['low'].shift(1) - 1
    df['close_return'] = df['close']/df['close'].shift(1) - 1
    df['log volume'] = np.log(df['volume'])
    df['log volatility'] = Volatility(df['close_return'].iloc[1:])
    df['log high-low'] = np.log(df['high'] - df['low'])
    df['open-close'] = (df['open'] - df['close'])
    df['EMA5'] = EMA(df['close'], 5)
    df['EMA10'] = EMA(df['close'], 10)
    df['EMA20'] = EMA(df['close'], 20)
    df['EMA30'] = EMA(df['close'], 30)
    df['STD5'] = ESTD(df['close_return'], 5)
    df['STD10'] = ESTD(df['close_return'], 10)
    df['STD20'] = ESTD(df['close_return'], 20)
    df['STD30'] = ESTD(df['close_return'], 30)
    df['BB high'], df['BB mid'], df['BB low'] = BBANDS(df['close'], 20, 2, 2)
    df['MACD fast'], df['MACD slow'], df['MACD'] = MACD(df['close'], 12, 26, 9)
    
    # normalize
    for col in featuresList:
        df[col] /= df[col].max(skipna=True)
    
    df.dropna(inplace = True)
    return df

def createWindow(df: pd.DataFrame):
    if len(df.index) < windowSize:
        return None
    x = df.iloc[-windowSize:]
    y = df['close'].iloc[-1]/df['close'].iloc[-2] - 1
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    
def addM0(data):
    for i, symbol in enumerate(symbolList):
        operandsValues['m0'][i] = data[symbol]
        
def predict():
    for operations in alpha.operations:
        executeOperation(operations)

def allocateWeights(predictArray: list):
    predictArray = pd.Series(predictArray)
    longIdx = predictArray.nlargest(noLongShort, keep = 'first').index.tolist()
    shortIdx = predictArray.nsmallest(noLongShort, keep = 'last').index.tolist()
    weight = 1/noLongShort
    
    weights = np.zeros(len(predictArray))
    weights[longIdx] = weight 
    weights[shortIdx] = -weight 
    return weights 

def getPosition(weights):
    currStocks = [ex.GetAccount()['Stocks'] for ex in exchanges]
    currBuy = [ex.GetTicker()['Buy'] for ex in exchanges]
    currSell = [ex.GetTicker()['Sell'] for ex in exchanges]
    
    targetStocks = initialStocks.copy()
    for i in range(len(symbolList)):
        if weights[i] > 0:
            targetStocks[i] += initialBalance*weights[i]/currSell[i]
        elif weights[i] < 0:
            targetStocks[i] += initialBalance*weights[i]/currBuy[i]
        else:
            targetStocks[i] += 0
    
    transactStocks = [targetStocks[i] - currStocks[i] for i in range(len(symbolList))]
    return transactStocks

def setOrder(transactStocks):
    currBuy = [ex.GetTicker()['Buy'] for ex in exchanges]
    currSell = [ex.GetTicker()['Sell'] for ex in exchanges]
    for i, ex in enumerate(exchanges):
        if transactStocks[i] > 0:
            ex.Buy((currBuy[i]+currSell[i])/2, abs(transactStocks[i]))
        elif transactStocks[i] < 0:
            ex.Sell((currBuy[i]+currSell[i])/2, abs(transactStocks[i]))

def onTick():
    data = {}
    for i, symbol in enumerate(symbolList):
        ex = exchanges[i]
        dat = getData(ex)
        dat = processData(dat)
        if dat is None: return
        dat, y = createWindow(dat)
        if dat is None:
            return
        else: 
            data[symbol] = dat
            #add S0
            operandsValues['s0'][i] = y
            
    addM0(data)
    predict()
    weights = allocateWeights(operandsValues['s1'])
    transactStocks = getPosition(weights)
    setOrder(transactStocks)
    
def main():
    while True:
        onTick()
        Sleep(int(86399200/24/6)) #sleep for 1 day


        
        
        
        
        
        
        
        
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

class inputError(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message
    
def OP1(s1: float, s2: float) -> np.float32:
    # s1 + s2
    return s1 + s2

def OP2(s1: float, s2: float) -> float:
    # s1 - s2
    return s1 - s2

def OP3(s1: float, s2: float) -> float:
    # s1 * s2
    return s1 * s2

def OP4(s1: float, s2: float) -> float:
    # s1 / s2
    if s2 == 0:
        raise (inputError("Divisor cannot be 0"))
    return s1 / s2

def OP5(s: float) -> float:
    # |s|
    return abs(s)

def OP6(s: float) -> float:
    # 1/s
    if s == 0:
        raise (inputError("Cannot inverse 0"))
    return 1 / s

def OP7(s: float) -> np.float32:
    # sin(s)
    return np.sin(s)

def OP8(s: float) -> np.float32:
    # cos(s)
    return np.cos(s)

def OP9(s: float) -> np.float32:
    # tan(s)
    return np.tan(s)

def OP10(s: float) -> np.float32:
    # arcsin(s)
    if abs(s) > 1:
        s = np.sign(s)
    return np.arcsin(s)

def OP11(s: float) -> np.float32:
    # arccos(s)
    if abs(s) > 1:
        s = np.sign(s)
    return np.arccos(s)

def OP12(s: float) -> np.float32:
    # arctan(s)
    return np.arctan(s)

def OP13(s: float) -> np.float32:
    # e^s
    return np.exp(s)

def OP14(s: float) -> np.float32:
    # ln(s)
    if s <= 0:
        raise (inputError("Input must be positive"))
    return np.log(s)

def OP15(s: float) -> np.float32:
    # heaviside(s) for float
    return max(np.sign(s), 0)

def OP16(v: np.ndarray) -> np.ndarray:
    # heaviside(v) for np.ndarray
    return np.heaviside(v, 0)

def OP17(m: np.ndarray) -> np.ndarray:
    # heaviside(m) for np.ndarray
    return np.heaviside(m, 0)

def OP18(s: float, v: np.ndarray) -> np.ndarray:
    # s*v
    return s * v

def OP19(s: float, i: int) -> np.ndarray:
    # v = bcast(s): float to np.ndarray
    # i: length of np.ndarray output
    return np.array([s] * i)

def OP20(v: np.ndarray) -> np.ndarray:
    # 1/v: inverse of np.ndarray
    if (v == 0).any():
        raise (inputError("Element cannot be 0"))
    return 1 / v

def OP21(v: np.ndarray) -> np.float32:
    # ||v||
    return np.linalg.norm(v)

def OP22(v: np.ndarray) -> np.ndarray:
    # |v|
    return abs(v)

def OP23(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # v1+v2
    if len(v1) != len(v2):
        raise inputError("np.ndarrays input must have the same size")
    return v1 + v2

def OP24(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # v1-v2
    if len(v1) != len(v2):
        raise inputError("np.ndarrays input must have the same size")
    return v1 - v2

def OP25(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # v1*v2
    if len(v1) != len(v2):
        raise inputError("np.ndarrays input must have the same size")
    return v1 * v2

def OP26(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # v1/v2
    if len(v1) != len(v2):
        raise inputError("np.ndarrays input must have the same size")
    if (v2 == 0).any():
        raise inputError("Divisor input cannot be 0")
    return v1 / v2

def OP27(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # dot product of 2 np.ndarrays
    if len(v1) != len(v2):
        raise inputError("np.ndarrays input must have the same size")
    return np.dot(v1, v2)

def OP28(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # outer product
    return np.outer(v1, v2)

def OP29(s: float, m: np.ndarray) -> np.ndarray:
    # s*m
    return s * m

def OP30(m: np.ndarray) -> np.ndarray:
    # 1/m
    if (m == 0).any():
        raise (inputError("Element cannot be 0"))
    return 1 / m

def OP31(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    # dot(m, v)
    if m.shape[1] != len(v):
        raise inputError("np.ndarray input must have the same size as row size of np.ndarray")
    return np.dot(m, v)

def OP32(v: np.ndarray, i: int) -> np.ndarray:
    # [[1, 2, 3, 4],
    # [1, 2, 3, 4],
    # [1, 2, 3, 4]]
    return np.array([v] * i)

def OP33(v: np.ndarray, j: int) -> np.ndarray:
    # [[1, 1, 1],
    # [2, 2, 2],
    # [3, 3, 3],
    # [4, 4, 4]]
    return np.transpose(np.array([v] * j))

def OP34(m: np.ndarray) -> np.float32:
    # ||m||
    return np.linalg.norm(m)

def OP35(m: np.ndarray) -> np.ndarray:
    # v[i] = norm(m[i,])
    return np.linalg.norm(m, axis=1)

def OP36(m: np.ndarray) -> np.ndarray:
    # v[j] = norm(m[,j])
    return np.linalg.norm(m, axis=0)

def OP37(m: np.ndarray) -> np.ndarray:
    # transpose(m)
    return np.transpose(m)

def OP38(m: np.ndarray) -> np.float32:
    # |m|
    return abs(m)

def OP39(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    # m1+m2
    if m1.shape != m2.shape:
        raise inputError("Matrices must have the same size")
    return m1 + m2

def OP40(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    # m1-m2
    if m1.shape != m2.shape:
        raise inputError("Matrices must have the same size")
    return m1 - m2

def OP41(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    # m1*m2
    if m1.shape != m2.shape:
        raise inputError("Matrices must have the same size")
    return m1 * m2

def OP42(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    # m1/m2
    if m1.shape != m2.shape:
        raise inputError("Matrices must have the same size")
    if (m2 == 0).any():
        raise (inputError("Element cannot be 0"))
    return m1 / m2

def OP43(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    # matmul(m1, m2)
    return np.matmul(m1, m2)

def OP44(s1: float, s2: float) -> float:
    # min(s1, s2)
    return min(s1, s2)

def OP45(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # min(v1, v2)
    if len(v1) != len(v2):
        raise inputError("np.ndarrays must have the same size")
    return np.minimum(v1, v2)

def OP46(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    # min(m1, m2)
    if m1.shape != m2.shape:
        raise inputError("Matrices must have the same shape")
    return np.minimum(m1, m2)

def OP47(s1: float, s2: float) -> float:
    # max(s1, s2)
    return (max(s1, s2))

def OP48(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # max(v1, v2)
    if len(v1) != len(v2):
        raise inputError("np.ndarrays must have the same size")
    return np.maximum(v1, v2)

def OP49(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    # max(m1, m2)
    if m1.shape != m2.shape:
        raise inputError("Matrices must have the same shape")
    return (np.maximum(m1, m2))

def OP50(v: np.ndarray) -> np.float32:
    # mean(v)
    return np.mean(v)

def OP51(m: np.ndarray) -> np.float32:
    # mean(m)
    return np.mean(m)

def OP52(m: np.ndarray) -> np.ndarray:
    # np.ndarray mean of each row
    return np.mean(m, axis=1)

def OP53(m: np.ndarray) -> np.ndarray:
    # np.ndarray std of each row
    return np.std(m, axis=1)

def OP54(v: np.ndarray) -> np.float32:
    # std of np.ndarray
    return np.std(v)

def OP55(m: np.ndarray) -> np.float32:
    # std of np.ndarray
    return np.std(m)

def OP56(const: float) -> float:
    # Initiate constant float
    return const

def OP57(const: float, i: int) -> np.ndarray:
    # Initiate constant np.ndarray
    return np.array([const] * i)

def OP58(const: float, i: int, j: int) -> np.ndarray:
    # Initiate constant np.ndarray
    return ([[const] * j] * i)

def OP59(a: float = -1, b: float = 1) -> float:
    # generate a random float from uniform(a, b)
    return np.random.uniform(low=min(a, b), high=max(a, b))

def OP60(a: float, b: float, i: int) -> np.ndarray:
    # generate a random np.ndarray from uniform(a, b)
    return np.random.uniform(low=min(a, b), high=max(a, b), size=(i,))

def OP61(a: float, b: float, i: int, j: int) -> np.ndarray:
    # generate a random np.ndarray from uniform(a, b)
    return np.random.uniform(low=min(a, b), high=max(a, b), size=(i, j))

def OP62(mean: float = 0, std: float = 1) -> float:
    # generate a random float from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return np.random.normal(loc=mean, scale=std)

def OP63(mean: float, std: float, i: int) -> np.ndarray:
    # generate a random np.ndarray from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return np.random.normal(loc=mean, scale=std, size=(i,))

def OP64(mean: float, std: float, i: int, j: int) -> np.ndarray:
    # generate a random np.ndarray from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return np.random.normal(loc=mean, scale=std, size=(i, j))

def OP65(array: list) -> list:
    array = np.array(array)
    array = rankdata(array)
    return list(array)

def OP66(array: list) -> list:
    array = np.array(array)
    array = rankdata(array)
    return list(array)

def OP67(array: list) -> list:
    array = np.array(array)
    array = array - sum(array)/len(array)
    return list(array)

OP_dict = {1: OP1, 
           2: OP2,
           3: OP3,
           4: OP4,
           5: OP5,
           6: OP6,
           7: OP7,
           8: OP8,
           9: OP9,
           10: OP10,
           11: OP11, 
           12: OP12,
           13: OP13,
           14: OP14,
           15: OP15,
           16: OP16,
           17: OP17,
           18: OP18,
           19: OP19,
           20: OP20,
           21: OP21, 
           22: OP22,
           23: OP23,
           24: OP24,
           25: OP25,
           26: OP26,
           27: OP27,
           28: OP28,
           29: OP29,
           30: OP30,
           31: OP31, 
           32: OP32,
           33: OP33,
           34: OP34,
           35: OP35,
           36: OP36,
           37: OP37,
           38: OP38,
           39: OP39,
           40: OP40,
           41: OP41, 
           42: OP42,
           43: OP43,
           44: OP44,
           45: OP45,
           46: OP46,
           47: OP47,
           48: OP48,
           49: OP49,
           50: OP50,
           51: OP51, 
           52: OP52,
           53: OP53,
           54: OP54,
           55: OP55,
           56: OP56,
           57: OP57,
           58: OP58,
           59: OP59,
           60: OP60,
           61: OP61, 
           62: OP62,
           63: OP63,
           64: OP64,
           65: OP65,
           66: OP66,
           67: OP67 }

def executeOperation(operation: list):
    Output, op, Inputs = operation 
    OP = OP_dict[op]
    if op <= 55 and op not in [19, 32, 33]:
        for i, symbol in enumerate(symbolList):
            if len(Inputs) == 1:
                operandsValues[Output][i] = OP(np.array(operandsValues[Inputs[0]][i]))
            elif len(Inputs) == 2:
                operandsValues[Output][i] = OP(np.array(operandsValues[Inputs[0]][i]), np.array(operandsValues[Inputs[1]][i]))
            elif len(Inputs) == 3:
                operandsValues[Output][i] = OP(np.array(operandsValues[Inputs[0]][i]), np.array(operandsValues[Inputs[1]][i]), np.array(operandsValues[Inputs[2]][i]))
            elif len(Inputs) == 4:
                operandsValues[Output][i] = OP(np.array(operandsValues[Inputs[0]][i]), np.array(operandsValues[Inputs[1]][i]), np.array(operandsValues[Inputs[2]][i]), np.array(operandsValues[Inputs[3]][i]))
    elif op in [19, 32, 33]:
        for i, symbol in enumerate(symbolList):
            operandsValues[Output][i] = OP(np.array(operandsValues[Inputs[0]][i]), Inputs[1])
    elif 56 <= op <= 64:
        for i, symbol in enumerate(symbolList):
            if len(Inputs) == 1:
                operandsValues[Output][i] = OP(Inputs[0])
            elif len(Inputs) == 2:
                operandsValues[Output][i] = OP(Inputs[0], Inputs[1])
            elif len(Inputs) == 3:
                operandsValues[Output][i] = OP(Inputs[0], Inputs[1], Inputs[2])
            elif len(Inputs) == 4:
                operandsValues[Output][i] = OP(Inputs[0], Inputs[1], Inputs[2], Inputs[3])
    elif 65 <= op <= 67:
        operandsValues[Output] = OP(operandsValues[Inputs[0]])
    

def EMA(series: pd.Series, window: int):
    return talib.EMA(series, timeperiod=window)


def ESTD(series: pd.Series, window: int):
    return series.ewm(window).std()


def SMA(series: pd.Series, window: int):
    return talib.SMA(series, timeperiod=window)


def STD(series: pd.Series, window: int):
    return talib.STDDEV(series, timeperiod=window)


def MACD(series: pd.Series, fastperiod: int, slowperiod: int, signalperiod: int):
    return talib.MACD(series, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)


def RSI(series: pd.Series, window: int):
    return talib.RSI(series, timeperiod=window)


def BBANDS(series: pd.Series, window: int, nbdevup: int, nbdevdn: int):
    return talib.BBANDS(series, timeperiod=window, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)


def Volatility(returns: pd.Series):
    new_returns = 100*returns # convert returns to %. For convergence purpose of the algorithm
    model = arch_model(new_returns, vol='GARCH', p=1, o=1, q=1, dist='normal')
    result = model.fit()
    return result.conditional_volatility
