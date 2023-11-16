from sympy import prevprime
from base import *


class UGM(Enum):

    LINEAR = 'Linear' # 简单线性同余算法
    MT = 'Mersenne_Twister' # python标准库使用该方法
    PCG = 'PCG64' # np.random.default_rng使用该算法
    LH = "Latin_Hypercube" # scipy库提供的拉丁立方体采样算法
    PD = 'Poisson_Disk' # scipy库提供的泊松盘采样算法
    HALTON_SCIPY = 'Halton_scipy' # scipy库提供的halton序列
    SOBOL_SCIPY = 'Sobol_scipy' # scipy库提供的sobol序列
    SOBOL_UNIT = 'Sobol_unit' # quantlib库提供的sobol序列，采用单位生成矩阵
    SOBOL_KUO = 'Sobol_Kuo' # quantlib库提供的sobol序列，采用Kuo生成矩阵
    SOBOL_KUO3 = 'Sobol_Kuo3' # quantlib库提供的sobol序列，采用Kuo3生成矩阵
    SOBOL_JACK = 'Sobol_Jaeckel' # quantlib库提供的sobol序列，采用Jaeckel生成矩阵
    SOBOL_LL = 'Sobol_LevitanLemieux' # quantlib库提供的sobol序列，采用LevitanLemieux生成矩阵
    SOBOL_JK7 ='Sobol_JoeKuoD7' # quantlib库提供的sobol序列，采用JoeKuoD7生成矩阵

    @staticmethod
    def get_dict_source() -> dict:
        return {UGM.LINEAR: "base",
                UGM.MT: "standard",
                UGM.PCG: "numpy",
                UGM.LH: "scipy",
                UGM.PD: "scipy",
                UGM.HALTON_SCIPY: "scipy",
                UGM.SOBOL_SCIPY: "scipy",
                UGM.SOBOL_UNIT: "quantlib",
                UGM.SOBOL_KUO: "quantlib",
                UGM.SOBOL_KUO3: "quantlib",
                UGM.SOBOL_JACK: "quantlib",
                UGM.SOBOL_LL: "quantlib",
                UGM.SOBOL_JK7: "quantlib"}


class Uniformer(Distribution):

    SAMPLE_RATIO = 1.0

    def __init__(self, scale:int=None, gen_type:UGM=None):
        super().__init__()
        self.points = {}
        self.discrepancy = {}

        if scale is not None and gen_type is not None:
            self.generate(scale, gen_type)

    def generate(self, scale:int, gen_type:UGM):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        gen_num = (int)(scale * Uniformer.SAMPLE_RATIO)
        if (gen_num, gen_type) in Uniformer.__objects__.keys():
            print("The object has been generated!(%d)[%s]" % (gen_num, gen_type.value))
            return

        self.__init__()
        start = time()
        self.gen_num = gen_num
        self.gen_type = gen_type
        num = 2 * (gen_num // 2) # num应为偶数

        if gen_type == UGM.LINEAR:
            ran = []
            for i in range(num):
                ran.append(math.modf((np.e * (i + RANDOM_SEED)))[0])
            self.sequences[SeqM.ORIGIN] = np.array(ran).reshape(-1)

        if gen_type == UGM.MT:
            ran = []
            for i in range(num):
                ran.append(random.random())
            self.sequences[SeqM.ORIGIN] = np.array(ran).reshape(-1)

        if gen_type == UGM.PCG:
            rng = np.random.default_rng(seed=RANDOM_SEED)
            self.sequences[SeqM.ORIGIN] = rng.random(num)

        if UGM.get_dict_source()[gen_type] == "scipy":
            num_points = num // 2
            max_batch_size = MAX_BATCH_SIZE // 2
            if gen_type == UGM.PD:
                rsg = stats.qmc.PoissonDisk(d=2, radius=(1 / np.sqrt(num_points)), ncandidates=(int)(np.sqrt(num_points)), seed=RANDOM_SEED, optimization='lloyd')
                self.points[SeqM.ORIGIN] = rsg.random(num_points)
            else:
                if gen_type == UGM.LH:
                    max_batch_size = prevprime((int)(math.sqrt(MAX_BATCH_SIZE // 2))) ** 2
                if gen_type == UGM.SOBOL_SCIPY:
                    max_batch_size = prevp2(MAX_BATCH_SIZE // 2)
                num_batch = 0
                while num_points > max_batch_size:
                    num_batch += 1
                    num_points -= max_batch_size
                if gen_type == UGM.LH:
                    num_points = prevprime((int)(math.sqrt(num_points // 2))) ** 2
                    rsg = stats.qmc.LatinHypercube(d=2, scramble=True, strength=2, seed=RANDOM_SEED, optimization='lloyd')
                if gen_type == UGM.HALTON_SCIPY:
                    rsg = stats.qmc.Halton(d=2, scramble=True, seed=RANDOM_SEED, optimization='lloyd')
                if gen_type == UGM.SOBOL_SCIPY:
                    num_points = prevp2(num_points // 2)
                    rsg = stats.qmc.Sobol(d=2, scramble=True, seed=RANDOM_SEED, optimization='lloyd')
                self.points[SeqM.ORIGIN] = np.zeros([num_points + max_batch_size * num_batch, 2])
                for i in range(num_batch):
                    self.points[SeqM.ORIGIN][i * max_batch_size: (i + 1) * max_batch_size] = rsg.random(max_batch_size)
                self.points[SeqM.ORIGIN][-num_points:] = rsg.random(num_points)
            self.sequences[SeqM.ORIGIN] = self.points[SeqM.ORIGIN].reshape(-1)

        if UGM.get_dict_source()[gen_type] == "quantlib":
            if gen_type == UGM.SOBOL_UNIT:
                rsg = ql.SobolRsg(MAX_BATCH_SIZE, RANDOM_SEED, ql.SobolRsg.Unit)
            if gen_type == UGM.SOBOL_KUO:
                rsg = ql.SobolRsg(MAX_BATCH_SIZE, RANDOM_SEED, ql.SobolRsg.Kuo)
            if gen_type == UGM.SOBOL_KUO3:
                rsg = ql.SobolRsg(MAX_BATCH_SIZE, RANDOM_SEED, ql.SobolRsg.Kuo3)
            if gen_type == UGM.SOBOL_JACK:
                rsg = ql.SobolRsg(MAX_BATCH_SIZE, RANDOM_SEED, ql.SobolRsg.Jaeckel)
            if gen_type == UGM.SOBOL_LL:
                rsg = ql.SobolRsg(MAX_BATCH_SIZE, RANDOM_SEED, ql.SobolRsg.SobolLevitanLemieux)
            if gen_type == UGM.SOBOL_JK7:
                rsg = ql.SobolRsg(MAX_BATCH_SIZE, RANDOM_SEED, ql.SobolRsg.JoeKuoD7)
            rsg.skipTo(SKIP_PATHS + RANDOM_SEED)
            self.sequences[SeqM.ORIGIN] = np.zeros(num)
            i = 0
            while num > MAX_BATCH_SIZE:
                self.sequences[SeqM.ORIGIN][i * MAX_BATCH_SIZE: (i + 1) * MAX_BATCH_SIZE] = np.array(rsg.nextSequence().value()).reshape(-1)
                i += 1
                num -= MAX_BATCH_SIZE
            self.sequences[SeqM.ORIGIN][i * MAX_BATCH_SIZE:] = np.array(rsg.nextSequence().value()).reshape(-1)[:num]

        if self.gen_type == None:
            raise ValueError("There's no such method!")
        self.gen_time = time() - start
        self.sequences[SeqM.DUAL] = np.append(self.sequences[SeqM.ORIGIN], 1 - self.sequences[SeqM.ORIGIN])
        np.random.shuffle(self.sequences[SeqM.DUAL])
        self.sequences[SeqM.STANDARD] = (1 - 1e-6) * (self.sequences[SeqM.DUAL] / np.max(self.sequences[SeqM.DUAL]))
        for st, s in self.sequences.items():
            self.points[st] = s.reshape(-1, 2)
        Uniformer.__objects__[(self.gen_num, self.gen_type)] = self

    def analyse(self, sampled:bool=False):
        if len(self.sequences) == 0:
            raise ValueError("Not yet generated!")
        self.sampled = sampled
        sequences = {}
        points = {}
        stats_size = 2 * (STATS_SIZE // 2) # stats_size应为偶数
        if sampled and len(self.sequences[SeqM.ORIGIN]) > stats_size:
            for st, s in self.sequences.items():
                sequences[st] = np.random.choice(s, stats_size, replace=False)
                points[st] = sequences[st].reshape(-1, 2)
        else:
            sequences = self.sequences
            points = self.points

        for st in SeqM:
            self.mean[st] = np.mean(sequences[st])
            self.var[st] = np.var(sequences[st])
            self.p_KStest[st] = stats.kstest(sequences[st], 'uniform')[1]
            self.p_CMtest[st] = stats.cramervonmises(sequences[st], 'uniform').pvalue
            if len(sequences[st]) > stats_size:
                self.discrepancy[st] = stats.qmc.discrepancy(np.random.choice(sequences[st], stats_size, replace=False).reshape(-1, 2), workers=1)
            else:
                self.discrepancy[st] = stats.qmc.discrepancy(sequences[st].reshape(-1, 2), workers=1)

    def report(self, save=False):
        if len(self.sequences) == 0:
            raise ValueError("Not yet generated!")
        if self.sampled is None:
            raise ValueError("Not yet analyzed!")
        print("[%s]UDtest Report" % self.gen_type.value)
        print("TimeCost: %f s" % self.gen_time)
        print("Num: %d" % len(self.sequences[SeqM.ORIGIN]))
        if self.sampled:
            print(" --sampled to %d" % STATS_SIZE)
        for st in SeqM:
            print("- Sequence Type: %s" % st.value)
            print("mean = %f, var = %f" % (self.mean[st], self.var[st]))
            print("KStest: p = %f" % self.p_KStest[st])
            print("CMtest: p = %f" % self.p_CMtest[st])
            print("Discrepancy: d = %f" % self.discrepancy[st])
        print("---END---", end="\n\n")
        if save:
            with open("reports\\UDtest\\(%d)[%s]Report" % (self.gen_num, self.gen_type.value), mode='w') as f:
                print("[%s]UDtest Report" % self.gen_type.value, file=f)
                print("TimeCost: %f s" % self.gen_time, file=f)
                print("Num: %d" % len(self.sequences[SeqM.ORIGIN]), file=f)
                if self.sampled:
                    print(" --sampled to %d" % STATS_SIZE, file=f)
                for st in SeqM:
                    print(
                        " -Sequence Type: %s" % st.value, file=f)
                    print("mean = %f, var = %f" % (self.mean[st], self.var[st]), file=f)
                    print("KStest: p = %f" % self.p_KStest[st], file=f)
                    print("CMtest: p = %f" % self.p_CMtest[st], file=f)
                    print("Discrepancy: d = %f" % self.discrepancy[st], file=f)
                print("---END---", end="\n\n", file=f)

if __name__ == '__main__':
    ign = [UGM.PD, UGM.LH, UGM.HALTON_SCIPY, UGM.SOBOL_SCIPY] # 需要忽略的方法

    for t in UGM:
        if t not in tuple(ign):
            re0 = Uniformer(BASE_SCALE, t)
            re0.analyse(sampled=False)
            re0.report(save=True)
    Uniformer.save()
