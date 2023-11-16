from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.stattools import omni_normtest, jarque_bera
from Uniformer import *


class NGM(Enum):

    BM_STD = 'Box_Muller_standard' # python标准库提供的Box_Muller算法
    ZGR_NUMPY = 'Ziggurat_numpy' # numpy库提供的Ziggurat算法
    ZGR_SCIPY = 'Ziggurat_scipy' # scipy库提供的Ziggurat算法
    ICDF = 'Inverse_CDF' # 反解CDF，由均匀分布生成正态分布的随机数
    CLT = 'Central_Limit_Theorem' # 根据中心极限定理，若干个独立均匀分布随机数的叠加可以生成正态分布的随机数
    BM = 'Box_Muller' # 采用Box_Muller算法，由均匀分布生成正态分布的随机数
    RS = 'Rejection_Sampling' # 对均匀分布作拒绝采样，得到正态分布的随机数
    # ZGR = 'Ziggurat' # ZGR正态分布生成算法，仍是拒绝采样的运用 [TODO: Not Complete]
    # 关于ZGR算法，可参考：
    # https://github.com/jameslao/zignor-python
    # https://en.wikipedia.org/wiki/Ziggurat_algorithm

    @staticmethod
    def get_dict_source() -> dict:
        return {NGM.BM_STD: "standard",
                NGM.ZGR_NUMPY: "numpy",
                NGM.ZGR_SCIPY: "scipy",
                NGM.ICDF: "base",
                NGM.CLT: "base",
                NGM.BM: "base",
                NGM.RS: "base"}


class Normalist(Distribution):

    SAMPLE_RATIO = 0.25

    def __init__(self, scale:int=None, gen_type:NGM=None, base_type:UGM=None, base_seq_type:SeqM=None):
        super().__init__()
        self.base_distribution = None
        self.base_seq_type = None
        self.skew = {} # 偏度
        self.kurt = {} # 峰度
        self.p_KSLtest = {} # Lilliefors test
        self.p_SKtest = {} # D'Agostino's K-squared test
        self.p_JBtest = {} # Jarque–Bera test

        if scale is not None and gen_type is not None:
            if NGM.get_dict_source()[gen_type] != "base":
                base_type = None
                base_seq_type = None
            self.generate(scale, gen_type, base_type, base_seq_type)

    def generate(self, scale:int, gen_type:NGM, base_type:UGM=None, base_seq_type:SeqM=None):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        gen_num = (int)(scale * Normalist.SAMPLE_RATIO)
        if NGM.get_dict_source()[gen_type] != "base":
            base_type = None
            base_seq_type = None
        if (gen_num, gen_type, base_type, base_seq_type) in Normalist.__objects__.keys():
            if base_type is not None:
                print("The object has been generated!(%d)[%s][%s][%s]" % (gen_num, gen_type.value, base_type.value, base_seq_type.value))
            else:
                print("The object has been generated!(%d)[%s]" % (gen_num, gen_type.value))
            return

        self.__init__()
        start = time()
        self.gen_num = gen_num
        self.gen_type = gen_type
        num = gen_num
        base_num = (int)(scale * Uniformer.SAMPLE_RATIO)

        if gen_type == NGM.BM_STD:
            self.sequences[SeqM.ORIGIN] = np.array([random.normalvariate(0, 1) for _ in range(num)])

        if gen_type == NGM.ZGR_NUMPY:
            self.sequences[SeqM.ORIGIN] = np.random.normal(loc=0, scale=1, size=num)

        if gen_type == NGM.ZGR_SCIPY:
            self.sequences[SeqM.ORIGIN] = stats.norm.rvs(loc=0, scale=1, size=num)

        if NGM.get_dict_source()[gen_type] == "base":
            if (base_num, base_type) in Uniformer.__objects__.keys():
                self.base_distribution = Uniformer.get_obj((base_num, base_type))
            else:
                self.base_distribution = Uniformer(base_num, base_type)
            self.base_seq_type = base_seq_type
            base_sequence = self.base_distribution.get_sequence(base_seq_type)

            if gen_type == NGM.ICDF:
                self.sequences[SeqM.ORIGIN] = stats.norm.ppf(np.random.choice(base_sequence, size=num, replace=False)) # 基于Newton迭代法
            if gen_type == NGM.CLT:
                self.sequences[SeqM.ORIGIN] = []
                for i in range(num):
                    self.sequences[SeqM.ORIGIN].append(np.random.choice(base_sequence, size=(int)(np.sqrt(num)), replace=True).mean())
            if gen_type == NGM.BM:
                base_sequence = np.random.choice(base_sequence, size=num, replace=False)
                self.sequences[SeqM.ORIGIN] = []
                for i in range(len(base_sequence) // 2):
                    r = np.sqrt(-2 * np.log(base_sequence[2 * i]))
                    theta = 2 * np.pi * base_sequence[2 * i + 1]
                    self.sequences[SeqM.ORIGIN].append(r * np.cos(theta))
                    self.sequences[SeqM.ORIGIN].append(r * np.sin(theta))
                if len(base_sequence) % 2 == 1:
                    self.sequences[SeqM.ORIGIN].append(0)
            if gen_type == NGM.RS:
                self.sequences[SeqM.ORIGIN] = []
                while len(self.sequences[SeqM.ORIGIN]) < num:
                    x = np.random.choice((base_sequence - 0.5) * 8, size=1)[0]
                    u = np.random.uniform(0, 1 / np.sqrt(2 * np.pi))
                    if u <= stats.norm.pdf(x):
                        self.sequences[SeqM.ORIGIN].append(x)
            self.sequences[SeqM.ORIGIN] = np.array(self.sequences[SeqM.ORIGIN])

        if self.gen_type == None:
            raise ValueError("There's no such method!")
        self.gen_time = time() - start
        self.sequences[SeqM.DUAL] = np.append(self.sequences[SeqM.ORIGIN], -1 * self.sequences[SeqM.ORIGIN])
        np.random.shuffle(self.sequences[SeqM.DUAL])
        self.sequences[SeqM.STANDARD] = stats.zscore(self.sequences[SeqM.DUAL])
        if NGM.get_dict_source()[gen_type] == "base":
            Normalist.__objects__[(self.gen_num, self.gen_type, self.base_distribution.gen_type, self.base_seq_type)] = self
        else:
            Normalist.__objects__[(self.gen_num, self.gen_type, None, None)] = self

    def analyse(self, sampled:bool=False):
        if len(self.sequences) == 0:
            raise ValueError("Not yet generated!")
        self.sampled = sampled
        sequences = {}
        if sampled and len(self.sequences[SeqM.ORIGIN]) > STATS_SIZE:
            for st, s in self.sequences.items():
                sequences[st] = np.random.choice(s, STATS_SIZE, replace=False)
        else:
            sequences = self.sequences

        for st in SeqM:
            self.mean[st] = np.mean(sequences[st])
            self.var[st] = np.var(sequences[st])
            self.p_KStest[st] = stats.kstest(sequences[st], 'norm')[1]
            self.p_CMtest[st] = stats.cramervonmises(sequences[st], 'norm').pvalue
            self.skew[st] = stats.skew(sequences[st])
            self.kurt[st] = stats.kurtosis(sequences[st], fisher=True)
            self.p_KSLtest[st] = lilliefors(sequences[st])[1]
            self.p_SKtest[st] = omni_normtest(sequences[st])[1]
            self.p_JBtest[st] = jarque_bera(sequences[st])[1]

    def report(self, save=False):
        if len(self.sequences) == 0:
            raise ValueError("Not yet generated!")
        if self.sampled is None:
            raise ValueError("Not yet analyzed!")
        print("[%s]NDtest Report" % self.gen_type.value)
        if self.base_distribution is not None:
            print("  --based on %s [%s]" % (self.base_distribution.gen_type.value, self.base_seq_type.value))
        print("TimeCost: %f s" % self.gen_time)
        print("Num: %d" % len(self.sequences[SeqM.ORIGIN]))
        if self.sampled:
            print("  --sampled to %d" % STATS_SIZE)
        for st in SeqM:
            print("- Sequence Type: %s" % st.value)
            print("mean = %f, var = %f" % (self.mean[st], self.var[st]))
            print("stew = %f, kurt = %f" % (self.skew[st], self.kurt[st]))
            print("KStest: p = %f" % self.p_KStest[st])
            print("CMtest: p = %f" % self.p_CMtest[st])
            print("KSLtest: p = %f" % self.p_KSLtest[st])
            print("SKtest: p = %f" % self.p_SKtest[st])
            print("JBtest: p = %f" % self.p_JBtest[st])
        print("---END---", end="\n\n")
        if save:
            if self.base_distribution is not None:
                path = "reports\\NDtest\\(%d)[%s][%s][%s]Report" % (self.gen_num, self.gen_type.value, self.base_distribution.gen_type.value, self.base_seq_type.value)
            else:
                path = "reports\\NDtest\\(%d)[%s]Report" % (self.gen_num, self.gen_type.value)
            with open(path, mode='w') as f:
                print("[%s]NDtest Report" % self.gen_type.value, file=f)
                if self.base_distribution is not None:
                    print("  --based on %s [%s]" % (self.base_distribution.gen_type.value, self.base_seq_type.value), file=f)
                print("TimeCost: %f s" % self.gen_time, file=f)
                print("Num: %d" % len(self.sequences[SeqM.ORIGIN]), file=f)
                if self.sampled:
                    print("  --sampled to %d" % STATS_SIZE, file=f)
                for st in SeqM:
                    print("- Sequence Type: %s" % st.value, file=f)
                    print("mean = %f, var = %f" % (self.mean[st], self.var[st]), file=f)
                    print("stew = %f, kurt = %f" % (self.skew[st], self.kurt[st]), file=f)
                    print("KStest: p = %f" % self.p_KStest[st], file=f)
                    print("CMtest: p = %f" % self.p_CMtest[st], file=f)
                    print("KSLtest: p = %f" % self.p_KSLtest[st], file=f)
                    print("SKtest: p = %f" % self.p_SKtest[st], file=f)
                    print("JBtest: p = %f" % self.p_JBtest[st], file=f)
                print("---END---", end="\n\n", file=f)


if __name__ == '__main__':
    Uniformer.read(BASE_SCALE)
    ign = [NGM.RS, NGM.CLT]

    for t in NGM:
        if t not in tuple(ign):
            if NGM.get_dict_source()[t] == "base":
                for key in Uniformer.__objects__.keys():
                    bt = key[1]
                    for st in SeqM:
                        re0 = Normalist(BASE_SCALE, t, bt, st)
                        re0.analyse(sampled=False)
                        re0.report(save=True)
            else:
                re0 = Normalist(BASE_SCALE, t)
                re0.analyse(sampled=False)
                re0.report(save=True)
    Normalist.save()
