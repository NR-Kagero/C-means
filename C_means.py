import numpy as np


class Cmeans:
    def __init__(self, clusters=2, g=2):
        self.__error = None
        self.__clusters_num = clusters
        self.__g = g
        self.__clusters_center = None
        self.__memberships = None

    def fit(self, x, num_iterations=100, thresh=0.01, seed=42, seed_enable=False):
        memberships = self._member_generate(x.shape[0],seed_enable=True)
        CC_old = self._feed(x, memberships)
        distances = self._distance(x, CC_old)
        error = thresh * 2
        i = 0
        CC_new = np.array(CC_old)
        n_memberships = np.array(memberships)

        while i < num_iterations and error > thresh:
            n_memberships = self._update_memberships(distances, n_memberships)
            CC_new = self._feed(x, n_memberships)
            error = self._threshold(CC_old, CC_new)
            distances = self._distance(x, CC_new)
            CC_old = np.array(CC_new)
            i = i + 1
        self.__clusters_center = CC_new
        self.__memberships = n_memberships
        self.__error = error

    def _member_generate(self, x,seed=42,seed_enable=False):
        if seed_enable:
            np.random.seed(seed)
        m = np.random.randint(5, 10, x*self.__clusters_num).reshape(-1, self.__clusters_num)
        m = m / np.sum(m, axis=1).reshape(-1, 1)
        m=np.array(m)
        return m


    def _feed(self, x, members):
        X = np.array(x).reshape(-1, len(x))
        print(X)
        print(members)
        print(members.sum(axis=1))
        centers = np.dot(X, members ** self.__g)
        centers = np.array(centers).reshape(-1, X.shape[0])
        print(centers)
        #os.system('cmd /k "pause"')
        print("kiko")
        centers = centers / np.sum(members ** self.__g, axis=0).reshape(-1, 1)
        return np.array(centers)

    def _distance(self, x, CC):
        dis = []
        for i in x:
            dum = []
            for I in CC:
                dum.append(np.sqrt(np.sum((i - I) ** 2)))
            dis.append(dum)
        dis=np.array(dis)
        return dis

    def _update_memberships(self, dis, members):
        new_members = np.array(members)
        for i in range(members.shape[0]):
            for I in range(members.shape[1]):
                new_members[i][I] = self._one_update(dis[i], I)
        return new_members

    def _one_update(self, line, ind):
        B=0.0
        for i in line:
            B+=(line[ind]/i)**(2/(self.__g-1))
        return 1/B

    def predict(self,data):
        distances=self._distance(data,self.__clusters_center)
        prediction=[x.index(max(x))+1 for x in distances]
        return np.array(prediction)

    def _threshold(self, CC_old, CC_new):
        c1 = CC_old.reshape(-1)
        c2 = CC_new.reshape(-1)
        return np.sum(abs(c1 - c2))

    def get_cluster_center(self):
        return self.__clusters_center

    def get_memberships(self):
        return self.__memberships

    def get_last_error(self):
        return self.__error
