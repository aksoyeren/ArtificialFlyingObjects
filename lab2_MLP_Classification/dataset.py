import numpy as np

class MLPData:
    """ """
    @staticmethod
    def syn1(N):
        """data(samples, features)

        :param N: 

        """
        data = np.empty(shape=(N,2), dtype = np.float32)  
        tar = np.empty(shape=(N,), dtype = np.float32) 
        N1 = int(N/2)

        data[:N1,0] = 4 + np.random.normal(loc=.0, scale=1., size=(N1))
        data[N1:,0] = -4 + np.random.normal(loc=.0, scale=1., size=(N-N1))
        data[:,1] = 10*np.random.normal(loc=.0, scale=1., size=(N))


        data = data / data.std(axis=0)

        # Target
        tar[:N1] = np.ones(shape=(N1,))
        tar[N1:] = np.zeros(shape=(N-N1,))

        # Rotation
        theta = np.radians(30)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c,-s],[s,c]]) # rotation matrix
        data = np.dot(data,R) 

        return data,tar
    
    @staticmethod
    def syn2(N):
        """data(samples, features)

        :param N: 

        """

        data = np.empty(shape=(N,2), dtype = np.float32)  
        tar = np.empty(shape=(N,), dtype = np.float32) 
        N1 = int(N/2)

        # Positive samples
        data[:N1,:] = 0.8 + np.random.normal(loc=.0, scale=1., size=(N1,2))
        # Negative samples 
        data[N1:,:] = -.8 + np.random.normal(loc=.0, scale=1., size=(N-N1,2))


        # Target
        tar[:N1] = np.ones(shape=(N1,))
        tar[N1:] = np.zeros(shape=(N-N1,))

        return data,tar
    
    @staticmethod
    def syn3(N):
        """data(samples, features)

        :param N: 

        """
        data = np.empty(shape=(N,2), dtype = np.float32)  
        tar = np.empty(shape=(N,), dtype = np.float32) 
        N1 = int(2*N/3)

        # disk
        teta_d = np.random.uniform(0, 2*np.pi, N1)
        inner, outer = 2, 5
        r2 = np.sqrt(np.random.uniform(inner**2, outer**2, N1))
        data[:N1,0],data[:N1,1] = r2*np.cos(teta_d), r2*np.sin(teta_d)

        #circle
        teta_c = np.random.uniform(0, 2*np.pi, N-N1)
        inner, outer = 0, 3
        r2 = np.sqrt(np.random.uniform(inner**2, outer**2, N-N1))
        data[N1:,0],data[N1:,1] = r2*np.cos(teta_c), r2*np.sin(teta_c)

        # Normalization
        #data = data - data.mean(axis=0)/data.std(axis=0)

        tar[:N1] = np.ones(shape=(N1,))
        tar[N1:] = np.zeros(shape=(N-N1,))

        return data, tar
    
    @staticmethod
    def spiral(spiral_path):
        """

        :param spiral_path: 

        """
        tmp = np.loadtxt(spiral_path)
        data, tar = tmp[:, :2], tmp[:, 2]

        return data, tar
    
    @staticmethod
    def vowels(file_name_train='ae.train', file_name_test='ae.test'):
        """

        :param file_name_train: Default value = 'ae.train')
        :param file_name_test: Default value = 'ae.test')

        """
        def pre_proc(file_name):
            """

            :param file_name: 

            """
            block = []
            x = []

            with open(file_name) as file:
                for line in file:    
                    if line.strip():
                        numbers = [float(n) for n in line.split()]
                        block.append(numbers)
                    else:
                        x.append(block)
                        block = []

            ################################
            x = [np.asarray(ar) for ar in x]    
            return x

        x_train = pre_proc(file_name_train)
        x_test = pre_proc(file_name_test)


        ############## LABELS###########
        chunk1 = list(range(30,270, 30))
        y_train = []
        person = 0

        for i, block in enumerate(x_train):
            if i in chunk1:
                person += 1
            y_train.extend([person]*block.shape[0])

        chunk2 = [31,35,88,44,29,24,40,50,29]
        chunk2 = np.cumsum(chunk2)
        y_test = []
        person = 0
        for i, block in enumerate(x_test):
            if i in chunk2:
                person += 1
            y_test.extend([person]*block.shape[0])

        x_train = np.vstack(x_train)
        x_test = np.vstack(x_test)

        ## Split into train, validation and test
        num_classes = 9
        y_train = np.eye(num_classes, dtype='uint8')[y_train]#keras.utils.to_categorical(y_train, num_classes)
        y_test = np.eye(num_classes, dtype='uint8')[y_test]#keras.utils.to_categorical(y_test, num_classes)

        from sklearn.model_selection import train_test_split

        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.4, random_state=42)

        return x_train, y_train, x_val, y_val, x_test, y_test
    
    @staticmethod
    def regr1(N, v=0):
        """data(samples, features)

        :param N: param v:  (Default value = 0)
        :param v: Default value = 0)

        """
        data = np.empty(shape=(N,6), dtype = np.float32)  

        uni = lambda n : np.random.uniform(0,1,n)
        norm = lambda n : np.random.normal(0,1,n)
        noise =  lambda  n : np.random.normal(0,1,n)


        for i in range(4):
            data[:,i] = norm(N)
        for j in [4,5]:
            data[:,j] = uni(N)

        tar =   2*data[:,0] + data[:,1]* data[:,2]**2 + np.exp(data[:,3]) + \
                5*data[:,4]*data[:,5]  + 3*np.sin(2*np.pi*data[:,5])
        std_signal = np.std(tar)
        tar = tar + v * std_signal * noise(N)

        return data, tar
    
