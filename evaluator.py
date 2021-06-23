from collections import defaultdict
import metrics as Metrics
from surprise import accuracy, KNNBaseline
from surprise.model_selection import train_test_split,LeaveOneOut
import seaborn as sns
import time
from pprint import pprint
from tqdm import tqdm

class Evaluator():
    def __init__(self,data):
        self.data = data
        self.trainSet,self.testSet,self.LOOX_trainSet,self.LOOX_testSet,self.LOOX_antitestSet, \
            self.full_trainSet,self.full_antitestSet = self.processData(data)
        self.models = {}
        self.metrics = {}

    def processData(self,data):
        print('preparing data...')
        eval = EvaluationData(data,False)
        return eval.trainSet,eval.testSet,eval.LOOX_trainSet,eval.LOOX_testSet,eval.LOOX_antitestSet, \
            eval.full_trainSet,eval.full_antitestSet

    def addModel(self,model,name):
        self.models[name] = model

    def eval_10_fold(self, model_name):
        LOO_10_fold = LeaveOneOut(10, random_state=42)

        metrics = {
            'MAE': [],
            'RMSE': [],
            'cHR': [],
            'HR': []
        }
        for xtrain, xtest in tqdm(LOO_10_fold.split(self.data)):
            self.models[model_name].fit(xtrain)
            predictions = self.models[model_name].test(xtest)
            LOOXfull_predictions = self.models[model_name].test(self.LOOX_antitestSet)
            LOOX_topN = self.getTopN(LOOXfull_predictions)

            metrics['MAE'].append(accuracy.mae(predictions, verbose=False))
            metrics['RMSE'].append(accuracy.rmse(predictions, verbose=False))
            metrics['HR'].append(Metrics.HitRate(LOOX_topN, predictions))
            metrics['cHR'].append(Metrics.CumulativeHitRate(LOOX_topN, predictions, 4.0))

        return metrics

    def evaluateModel(self,doTopN=False):
        for name in self.models:
            t = time.time()
            print('Evaluating',name)
            self.models[name].fit(self.trainSet)
            predictions = self.models[name].test(self.testSet)
            # predictions_old = self.getPredicts(self.trainSet,self.testSet,fit)

            metrics = {}

            # Metrics: Accuracy
            metrics['MAE'] = accuracy.mae(predictions, verbose=False)
            metrics['RMSE'] = accuracy.rmse(predictions, verbose=False)

            if doTopN:
                self.models[name].fit(self.LOOX_trainSet)
                LOOX_predictions = self.models[name].test(self.LOOX_testSet)
                LOOXfull_predictions = self.models[name].test(self.LOOX_antitestSet)

                # self.models[name].fit(self.full_trainSet)
                # full_predictions = self.models[name].test(self.full_antitestSet)

                topN = self.getTopN(predictions)
                LOOX_topN = self.getTopN(LOOXfull_predictions)
                # full_topN = self.getTopN(full_predictions)

                # Metrics: Beyond, LOOX

                metrics['HR'] = Metrics.HitRate(LOOX_topN, LOOX_predictions)
                metrics['cHR'] = Metrics.CumulativeHitRate(LOOX_topN, LOOX_predictions, 4.0)
                metrics['ARHR'] = Metrics.ARHR(LOOX_topN, LOOX_predictions)

                # Metrics: Beyond, full dataset

                # metrics['Spread'] = Metrics.Spread(topN, full_predictions)
                # metrics['Coverage'] = Metrics.Coverage(full_predictions, self.full_antitestSet)
                # metrics['Diversity'] = Metrics.Diversity(full_topN, self.simAlgo)

            pprint(metrics)
            print('Time:',time.time() - t)
            print('-----------')
            self.metrics[name] = metrics

    def getPredicts(self,trainset,testset,model):
        predictions = []
        counter = 0
        total = 0
        for (uid, iid, true_r) in testset:
            total += 1
            try:
                res = (uid, iid, true_r, model.estimate(trainset.to_inner_uid(uid), trainset.to_inner_iid(iid)))
                predictions.append(res)
                counter += 1
            except Exception:
                # print('Could not get prediction:', uid, iid)
                continue
        print('Predictions made:',counter,'of',total)
        return predictions

    def sampleUser(self,testUser = '56'):
        self.testUserSummary(testUser)
        for name in self.models:
            print(name)
            self.models[name].fit(self.full_trainSet)
            predictions = self.getUserPredicts(self.models[name],self.full_trainSet,testUser)
            topN = self.getTopN(predictions)
            print("User's Top Recommendations:")
            for item,rating in topN[int(testUser)]:
                print(int(item),rating)
            print('-----------')

    def testUserSummary(self,testUser,isPlot=True):
        print('-----------')
        testUserInnerID = self.full_trainSet.to_inner_uid(testUser)
        print("User",testUser,"Total Ratings:", len(self.full_trainSet.ur[testUserInnerID]))
        print("User",testUser,"5 Star Ratings:")
        for iid, rating in self.full_trainSet.ur[testUserInnerID]:
            if rating == 5.0:
                print(int(self.full_trainSet.to_raw_iid(iid)))
                # print(ml.getMovieName(int(self.full_trainSet.to_raw_iid(iid))))
        print('-----------')
        if isPlot:
            sns.countplot([j for (_,j) in self.full_trainSet.ur[testUserInnerID]])
        return self

    def getUserPredicts(self,model,trainset,testUser):
        uid = trainset.to_inner_uid(testUser)
        anti_testset = []
        fill = trainset.global_mean
        user_items = set([j for (j,_) in trainset.ur[uid]])
        anti_testset += [(testUser,trainset.to_raw_iid(i),fill) for i in trainset.all_items() \
                         if i not in user_items]
        predictions = []
        counter = 0
        total = 0
        for (user, item, true_r) in anti_testset:
            total += 1
            try:
                res = (testUser,item,true_r,model.estimate(uid,trainset.to_inner_iid(item)))
                predictions.append(res)
                counter += 1
            except Exception:
                # print('Could not get prediction:', uid, iid)
                continue

        print('User predictions made:',counter,'of',total)
        return predictions

    def getTopN(self,predictions, n=10, minRating=3.0):
        topN = defaultdict(list)
        for pred in predictions:
            user, item, actualRating, predictRating = pred[0],pred[1],pred[2],pred[3]
            if predictRating >= minRating:
                topN[int(user)].append((int(item), predictRating))

        for user, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(user)] = ratings[:n]
        return topN

class EvaluationData:
    def __init__(self,data,withSim=False):
        self.trainSet, self.testSet = train_test_split(data, test_size=0.25, random_state=42)

        LOOX = LeaveOneOut(1, random_state=42)
        for xtrain, xtest in LOOX.split(data):
            self.LOOX_trainSet = xtrain
            self.LOOX_testSet = xtest
            del xtrain, xtest
        self.LOOX_antitestSet = self.LOOX_trainSet.build_anti_testset()

        self.full_trainSet = data.build_full_trainset()
        self.full_antitestSet = self.full_trainSet.build_anti_testset()
        if withSim:
            sim_options = {'name': 'cosine', 'user_based': False}
            self.simAlgo = KNNBaseline(sim_options=sim_options)
            self.simAlgo.fit(self.full_trainSet)
        