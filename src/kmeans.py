import random
import matplotlib.pyplot as plt
import numpy as np

class Kmeans():
    def __init__(self):
        self.to_pass_centers = []
        self.covariance = []
    def assign_clusters(self,X,Y,centers):
        n = len(X)
        m = len(centers)
        cluster_points = []
        for i in range(m):
            cluster_points.append({"X": [], "Y": [], "count": 0})
        for x, y in zip(X, Y):
            euc = 100000
            final_center = None
            i = 0
            for center in centers:
                X_center = center["X"]
                Y_center = center["Y"]
                dif = ((x - X_center) ** 2) + ((y - Y_center) ** 2)
                if euc > dif:
                    euc = dif
                    final_center = center
                    final_label = i
                i+=1
            cluster_points[final_label]["X"].append(x)
            cluster_points[final_label]["Y"].append(y)
            cluster_points[final_label]["count"] += 1
        return cluster_points


    def clustering(self,X,Y,cluster_centers):
        sumx = 0
        sumy = 0
        n = len(X)
        m = len(cluster_centers)
        clusters = []
        point_cluster_centers = []
        cluster_points = []
        all_cluster_points = []
        for i in range(m):
            cluster_points.append({"X":[],"Y":[],"count":0})
        for x,y in zip(X,Y):
            euc = 100000
            point_cluster_center = {}
            i=0
            for center in cluster_centers:
                dif = ((x - center["X"]) ** 2) + ((y - center["Y"]) ** 2)
                if euc > dif:
                    euc = dif
                    final_center = center
                    final_label = i
                i+=1

            #ith point ka kaunsa center hain aur kya label hain
            point_cluster_center["X"]=final_center["X"]
            point_cluster_center["Y"]=final_center["Y"]
            point_cluster_center["label"] = final_label
            #ith cluster main kaunse points hain aur kitna count hain
            cluster_points[final_label]["X"].append(x)
            cluster_points[final_label]["Y"].append(y)
            cluster_points[final_label]["count"] += 1
            point_cluster_centers.append(point_cluster_center)
        #naye points ka clusters
        final_cluster = []
        for i in range(len(cluster_centers)):
            final_cluster.append({"X":0,"Y":0})

        for i in range(len(cluster_centers)):
            final_cluster[i]["X"] = sum(cluster_points[i]["X"])/len(cluster_points[i]["X"])
            final_cluster[i]["Y"] = sum(cluster_points[i]["Y"])/len(cluster_points[i]["Y"])


        sse = 0
        for i in range(len(point_cluster_centers)):
            sse+=((point_cluster_centers[i]["X"]-X[i])**2+(point_cluster_centers[i]["Y"]-Y[i])**2)/len(point_cluster_centers)
        return final_cluster,point_cluster_centers,cluster_points,sse

    def calculate_diffence(self,previous_clusters,new_clusters):
        n = len(previous_clusters)
        bool = False
        for i in range(n):
            if previous_clusters[i]["X"]==new_clusters[i]["X"] and previous_clusters[i]["Y"]==new_clusters[i]["Y"]:
                continue
            else:
                bool = True
                break
        return bool

    def run_kmeans(self,k=3,r=2):
        file_object=open("Dataset_2.txt","r")

        #reading the entire txt file line by line
        content= file_object.readlines()
        X = []
        Y = []
        for x in content:
            line = x.split()
            X.append(float(line[0].strip()))
            Y.append(float(line[1].strip()))

        colorlist = ["blue","green","red","cyan","yellow","magenta","black","orange"]
        all_lses = []
        all_best_lses = []
        all_final_clusters = []
        all_cluster_points = []
        for c in range(r):
            centers = []
            for i in range(k):
                x = random.randint(0,len(content))
                centers.append({"X":X[x],"Y":Y[x],"label":i})
            clusters = []
            lse = []
            final_clusters, point_wise_label,cluster_points,sse = self.clustering(X,Y,centers)
            lse.append(sse)
            i=1
            while self.calculate_diffence(final_clusters,centers):
                centers = final_clusters
                final_clusters,point_wise_label,cluster_points,sse = self.clustering(X,Y,centers)
                lse.append(sse)
                #print("SSE: ",str(sse))
            all_best_lses.append(lse[len(lse)-1])
            all_final_clusters.append(final_clusters)
            all_cluster_points.append(cluster_points)
            all_lses.append(lse)
            #self.covariance.append(np.cov())
            plt.show()
            x=1/k
            # for c in range(len(cluster_points)):
            #     plt.plot(cluster_points[c]["X"],cluster_points[c]["Y"],'bo',color = colorlist[c%len(colorlist)])
            # plt.show()


        #plt.plot(lse_x,lse)
        best_r = -1
        minsse = 100000000
        count = 0
        for l in all_lses:

            lse_x = []
            s = l[len(l)-1]
            if s<minsse:
                minsse = s
                best_r = count
            for z in range(len(lse)):
                 lse_x.append(z)
            plt.plot(l)
            count+=1
        to_pass_centers = []
        for x in all_final_clusters[best_r]:
            temp = []
            temp.append(x["X"])
            temp.append(x["Y"])
            to_pass_centers.append(temp)
        for x in all_cluster_points[best_r]:
                self.covariance.append(np.cov(x["X"],x["Y"]))
        self.to_pass_centers = np.asarray(to_pass_centers)
        print("final lses:",str(all_best_lses))
        print("Best r:",str(r))
        plt.title("Change in square error over iterations")
        plt.xlabel("number of iterations to convergence")
        plt.ylabel("Squared Sum Error")
        plt.show()
        x_axis = []
        for x in range(len(all_best_lses)):
            x_axis.append(x)
        #plt.scatter(x_axis,all_best_lses)
        fig, ax = plt.subplots()
        ax.scatter(x_axis,all_best_lses)
        ax.plot(x_axis,all_best_lses)
        #plt.plot(x_axis,all_best_lses)
        for i, txt in enumerate(all_best_lses):
            ax.annotate(txt, (x_axis[i], all_best_lses[i]))
        plt.ylabel("Values of Square error at the convergence")
        plt.xlabel("Values of R")
        plt.title("Change in Square Error for diffrent random initialization")
        plt.show()
        for c in range(len(all_cluster_points[best_r])):
            plt.plot(all_cluster_points[best_r][c]["X"],all_cluster_points[best_r][c]["Y"],'bo',color = colorlist[c%len(colorlist)])
        plt.title("least squre error clustering for r = "+str(best_r+1))
        plt.xlabel("Y-axis")
        plt.ylabel("X-axis")
        plt.show()


if __name__ == "__main__":
    k = int(input("Enter k: "))
    r = int(input("Enter r:"))
    x=Kmeans()
    x.run_kmeans(k,r)