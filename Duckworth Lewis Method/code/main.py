import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

class DuckworthLewis:
    def __init__(self, data):
        self.data = data
    def filter(self):
        print("filtering...")
        self.matchIds = data.query("Innings == 1")
        self.matchIds = self.matchIds.groupby('Match').agg({"Over": "count", "Wickets.in.Hand": "min", "Match": "min"})
        self.matchIds = self.matchIds.query("Over == 50 or `Wickets.in.Hand` == 0")
        self.matchIds = self.matchIds["Match"].to_numpy().tolist()
        self.data = self.data.query("Innings == 1 and Match in(@self.matchIds)")
    def dataSelection(self):
        print("data selection...")
        tData = [ ]
        self.dataIndex = { "Match": 0, "Over": 1, "Runs": 2, "Wickets.in.Hand": 3, "Total.Runs": 4, "Innings.Total.Runs": 5, "Runs.Remaining": 6 }
        # Match, Over, Runs; Wickets.in.Hand; Total.Runs, Innings.Total.Runs, Runs.Remaining
        for i in range(len(self.data)):
            tData.append(self.data.iloc[i, [0, 3, 4, 11]].to_numpy().tolist())
        self.data = tData
    def dataCleaning(self):
        print("data cleaning...")
        # Total.Runs
        totRuns = { }
        for i, v in enumerate(self.data):
            if v[0] not in totRuns:
                totRuns[v[0]] = v[self.dataIndex["Runs"]]
            else:
                totRuns[v[0]] = totRuns[v[0]] + v[self.dataIndex["Runs"]]
            self.data[i].append(totRuns[v[0]])
        # Innings.Total.Runs
        for i, v in enumerate(self.data):
            self.data[i].append(totRuns[v[0]])
        # Runs.Remaining
        for i, v in enumerate(self.data):
            totRuns[v[0]] = totRuns[v[0]] - v[self.dataIndex["Runs"]]
            self.data[i].append(totRuns[v[0]])
    def initZ0(self):
        print("initializing Z0 values...")
        wVal = [ [ ] for i in range(11) ]
        tWickR = -1
        tMatch = -1
        tRunRem = -1
        for i in self.data:
            if tMatch != i[self.dataIndex["Match"]]:
                if tWickR > -1:
                    wVal[tWickR].append(tRunRem)
                    # print(tRunRem)
                tMatch = i[self.dataIndex["Match"]]
                tWickR = i[self.dataIndex["Wickets.in.Hand"]]
                tRunRem = i[self.dataIndex["Runs.Remaining"]]
            elif tWickR != i[self.dataIndex["Wickets.in.Hand"]]:
                wVal[tWickR].append(tRunRem)
                # print(tRunRem)
                tWickR = i[self.dataIndex["Wickets.in.Hand"]]
                tRunRem = i[self.dataIndex["Runs.Remaining"]]
            else:
                tRunRem = max(tRunRem, i[self.dataIndex["Runs.Remaining"]])
        wVal = [ sum(i) / len(i) for i in wVal ]
        self.Z0 = wVal[1 : 11]
        print("Initial Z0:")
        self.printZ0(self.Z0)
    def optimizeParams(self):
        print("mnimizing MSE... (may take about 1min)")
        finalMSE = 0
        WICKETS_IN_HAND = 3
        OVER = 1
        RUNS_REMAINING = 6
        def func(u, L, z0):
            return z0 * (1 - np.exp(-L * u / z0))
        def E(optim):
            global finalMSE
            L = optim[0]
            z0 = optim[1 : 11]
            mse = [ ]
            for i in self.data:
                if i[WICKETS_IN_HAND] > 0:
                    mse.append((func(50 - i[OVER], L, z0[i[WICKETS_IN_HAND] - 1]) - i[RUNS_REMAINING]) ** 2)
            finalMSE = sum(mse) / len(mse)
            return sum(mse) / len(mse)
        optims = [ 15 ] + self.Z0
        opt = optimize.minimize(E, optims, method='L-BFGS-B')
        self.L = opt.x[0]
        self.Z0 = opt.x[1 : 11]
        print(f"MSE: {finalMSE}")
        print(f"L: {self.L}")
        print("final values of Z0: ")
        self.printZ0(self.Z0)
        return { "L": self.L, "Z0": self.Z0, "MSE": finalMSE }
    def plotRunsObtainable(self):
        x1 = np.arange(50)
        legends = [ ]
        def func(u, L, z0):
            return z0 * (1 - np.exp(-L * u / z0))
        for w_rem in range(10):
            y1 = func(x1, self.L, self.Z0[w_rem])
            plt.plot(x1, y1)
            legends.append(str(w_rem + 1))
        plt.legend(legends)
        plt.title("agerage Runs obtainable")
        plt.xlabel("overs remaining")
        plt.ylabel("agerage Runs obtainable")
        plt.grid()
        plt.show()
    def printZ0(self, z0):
        str1 = ""
        for i in range(117):
            str1 = str1 + "-"
        print(str1)
        str1 = ""
        str1 = str1 + "| wicket in hand |"
        for i in range(1, 11):
            str1 = str1 + (f" {i}")
            for j in range(6 - len(str(i))):
                str1 = str1 + (" ")
            str1 = str1 + (" | ")
        print(str1)
        str1 = ""
        str1 = str1 + "| Z0             |"
        for i in z0:
            str1 = str1 + (" {:.2f}".format(i))
            for j in range(6 - len("{:.2f}".format(i))):
                str1 = str1 + (" ")
            str1 = str1 + (" | ")
        print(str1)
        str1 = ""
        for i in range(117):
            str1 = str1 + "-"
        print(str1)
data = pd.read_csv("../data/04_cricket_1999to2011.csv")
dw = DuckworthLewis(data)
dw.filter()
dw.dataSelection()
dw.dataCleaning()
dw.initZ0()
dw.optimizeParams()
dw.plotRunsObtainable()