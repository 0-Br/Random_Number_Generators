import os
from Normalist import *


class MonteCarlo:

    def __init__(self, S:float, sigma:float, K:float, T:float, r:float, rsg:Normalist=None, st:SeqM=None):
        self.S = S # 标的资产价格
        self.sigma = sigma # 波动率
        self.K = K # 行权价格
        self.T = T # 期权到期日，单位为年
        self.r = r # 无风险利率
        self.rsg = rsg # 正态分布随机数发生器
        self.st = st # 采用的序列类型
        self.I = None # 算法模拟次数

        self.re0_MC = None # 蒙特卡洛算法的模拟结果
        self.re0_BS = None # BS公式给出的估计结果
        self.error = None # 蒙特卡洛算法的误差

        self.valuate_BS()
        if self.rsg is not None and self.st is not None:
            self.valuate_MC(len(self.rsg.sequences[self.st]))

    def valuate_MC(self, I:int):
        self.I = I
        Z = np.random.choice(self.rsg.sequences[self.st], I, replace=False)
        ST = self.S * np.exp((self.r - (self.sigma ** 2) / 2) * self.T + self.sigma * np.sqrt(self.T) * Z) # 假设标的资产价格服从几何布朗运动
        HT = np.maximum(ST - self.K, 0) # 到期的回报，至少为正
        self.re0_MC = np.exp(-self.r * self.T) * np.mean(HT) # 记录蒙特卡洛算法的模拟结果
        if self.re0_BS is not None: # 计算误差
            self.error = abs((self.re0_BS - self.re0_MC) / self.re0_BS)
        return self.re0_MC

    def valuate_BS(self):
        d1 = (np.log(self.S / self.K) + (self.r + (self.sigma ** 2) / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S / self.K) + (self.r - (self.sigma ** 2) / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.re0_BS = self.S * stats.norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * stats.norm.cdf(d2)
        if self.re0_MC is not None: # 计算误差
            self.error = abs((self.re0_BS - self.re0_MC) / self.re0_BS)
        return self.re0_BS

    def report(self, save=False):
        if self.error is None:
            raise ValueError("Not yet started!")
        print("[%s]MonteCarlo Report" % self.rsg.gen_type.value)
        if self.rsg.base_distribution is not None:
            print("  --based on %s [%s]" % (self.rsg.base_distribution.gen_type.value, self.rsg.base_seq_type.value))
        print("[S=%.2f, sigma=%.2f, K=%.2f, T=%.2f, r=%.2f]"% (self.S, self.sigma, self.K, self.T, self.r))
        print("Eval: %.8f" % self.re0_MC)
        print("True: %.8f" % self.re0_BS)
        print("Error = %.6f" % self.error)
        print("---END---", end="\n\n")
        if save:
            dirpath = "reports\\MonteCarlo\\[S=%.2f, sigma=%.2f, K=%.2f, T=%.2f, r=%.2f]" % (self.S, self.sigma, self.K, self.T, self.r)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            if self.rsg.base_distribution is not None:
                path = dirpath + "\\(%d)[%s][%s][%s]Report" % (self.rsg.gen_num, self.rsg.gen_type.value, self.rsg.base_distribution.gen_type.value, self.rsg.base_seq_type.value)
            else:
                path = dirpath + "\\(%d)[%s]Report" % (self.rsg.gen_num, self.rsg.gen_type.value)
            with open(path, mode='w') as f:
                print("[%s]MonteCarlo Report" % self.rsg.gen_type.value, file=f)
                if self.rsg.base_distribution is not None:
                    print("  --based on %s [%s]" % (self.rsg.base_distribution.gen_type.value, self.rsg.base_seq_type.value), file=f)
                print("[S=%.2f, sigma=%.2f, K=%.2f, T=%.2f, r=%.2f]" % (self.S, self.sigma, self.K, self.T, self.r), file=f)
                print("Eval: %.8f" % self.re0_MC, file=f)
                print("True: %.8f" % self.re0_BS, file=f)
                print("Error = %.6f" % self.error, file=f)
                print("---END---", end="\n\n", file=f)


if __name__ == '__main__':
    Uniformer.read(BASE_SCALE)
    Normalist.read(BASE_SCALE)

    # 测试样例
    mc_examples = {(50, 0.3, 60, 1.0, 0.06): {},
                    (120, 0.15, 100, 0.25, 0.06): {},
                    (70, 0.35, 80, 1.5, 0.02): {},
                    (110, 0.4, 100, 1.2, 0.01): {},
                    (130, 0.3, 120, 0.9, -0.12): {},
                    (140, 0.25, 50, 1.3, 0.05): {},
                    (250, 2.0, 300, 0.8, 0.02): {},
                    (80, 0.4, 100, 20.0, 0.1): {},
                    (70, 0.35, 50, 1.5, 0.02): {},
                    (1000, 0.8, 1200, 5.0, 0.1): {}}

    for example in mc_examples.keys():
        for key in Normalist.__objects__.keys():
            t = key[1]
            bt = key[2]
            base_st = key[3]
            rsg = Normalist.get_obj(((int)(Normalist.SAMPLE_RATIO * BASE_SCALE), t, bt, base_st))
            for st in SeqM:
                mc = MonteCarlo(example[0], example[1], example[2], example[3], example[4], rsg, st)
                mc.report(save=True)
                mc_examples[example][(t, bt, base_st, st)] = (mc.error, mc.re0_MC, mc.re0_BS)

    with open("cache\\(%d)MontCarlo" % BASE_SCALE, "wb") as f:
        dump(mc_examples, f)
