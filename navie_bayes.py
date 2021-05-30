import pandas as pd


class navieBayes:
    def split_train_test(self,df):
        num_of_rows = df.shape[0]
        number_for_80_persent = int((8/10)*num_of_rows)
        train_x = df.iloc[:number_for_80_persent, :len(df.columns)-1]
        train_y = df.iloc[:number_for_80_persent, len(df.columns)-1]

        test_x = df.iloc[number_for_80_persent:, :len(df.columns)-1]
        test_y = df.iloc[number_for_80_persent:, len(df.columns)-1]

        return train_x, train_y, test_x, test_y

    def transform(self,trainx, trainy):
        pass

    def tell_total_types(self,list_of_record):
        temp_diff_val = []
        for value in list_of_record:
            if value not in temp_diff_val:
                temp_diff_val.append(value)
        return temp_diff_val
    def cal_probility_of_class(self,list_of_class_label):
        total_numbers = []
        for i in range(len(self.num_of_classes)):
            total_numbers.append(0)

        index = 0
        for Class in self.num_of_classes:
            for element in list_of_class_label:
                if Class == element:
                    total_numbers[index] = total_numbers[index] + 1
            index = index + 1

        probailty_of_classes = {}
        index = 0
        for Class in self.num_of_classes:
            probailty_of_classes[Class] = total_numbers[index] / len(list_of_class_label)
            index = index + 1
        return probailty_of_classes, total_numbers

    def fit(self, trainx, trainy):
        self.col_list = trainx.columns
        self.num_of_classes = []
        self.num_of_classes = self.tell_total_types(train_y)
        #print(self.num_of_classes)
        self.probabilty_of_class_var, self.count_of_class_var = self.cal_probility_of_class(train_y)
        self.class_count = {}
        for i in range(len(self.num_of_classes)):
            self.class_count[self.num_of_classes[i]] = self.count_of_class_var[i]

        #print(self.class_count)


        #print(self.probabilty_of_class_var)
        #print(self.count_of_class_var)

        self.list_of_train_ans = {}

        index = 0
        index_atrb = 0
        value = 0
        for Class in self.num_of_classes:
            #print("=*=*=*=*=*=*=*=*=*=*   ::  : ", Class, " :  ::  =*=*=*=*=*=*=*=*=*=*")
            col_dic = {}
            for col_name in train_x:
                #print("===================== : ", col_name, " : =================")
                col_list = train_x[col_name].tolist()
                #print("Col_list : ", col_list)
                self.dif_val = self.tell_total_types(col_list)
                #print("********************")
                #print("different Values : ",self.dif_val)
                val_dic = {}
                for val in self.dif_val:
                    for (atribute, Class_) in zip(col_list, train_y):
                        if atribute == val and Class == Class_:
                            value += 1
                    #print("Count with  : ", val, value)
                    value = value / self.class_count[Class]
                    val_dic[val] = value
                    value = 0
                col_dic[col_name] = val_dic
                #print("********************")
            index_atrb +=1
            self.list_of_train_ans[Class] = col_dic
            #print("\n\n\n\n")
        #print("_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=\n\n\n")
        #print(self.list_of_train_ans['yes'])
        #print("_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=\n\n\n")
        #print("_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=\n\n\n")
        #print(self.list_of_train_ans['no'])
        #print("_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=\n\n\n")
    def prdict(self,list_input):
        index = 0
        list_ans = {}
        value_compute = 1
        for key in self.list_of_train_ans:
            for keys in self.list_of_train_ans[key]:
                #print(self.list_of_train_ans[key][keys])
                for values in self.list_of_train_ans[key][keys]:
                    if values == list_input[index]:
                        value_compute *= self.list_of_train_ans[key][keys][values]
                index += 1
            list_ans[key] = value_compute
            value_compute = 1
            index = 0
        return max(list_ans, key=list_ans.get)
    def find_accuracy(self,y_expect, y_predict):
        match = 0
        for (exp,pre) in zip(y_expect,y_predict):
            if exp == pre:
                match += 1
        return (match/len(y_expect)*100)

    def predict_onlist(self,data):
        index = 0
        list_ans = {}
        value_compute = 1
        list_s = []
        list_of_result = []
        for ii, row in data.iterrows():
            for j, column in row.iteritems():
                list_s.append(column)
            for key in self.list_of_train_ans:
                for keys in self.list_of_train_ans[key]:
                    # print(self.list_of_train_ans[key][keys])
                    for values in self.list_of_train_ans[key][keys]:
                        if values == list_s[index]:
                            value_compute *= self.list_of_train_ans[key][keys][values]
                    index += 1
                list_ans[key] = value_compute
                value_compute = 1
                index = 0
            list_of_result.append(max(list_ans, key=list_ans.get))
            list_s = []
        return list_of_result



df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
nav_by = navieBayes()
train_x, train_y, test_x, test_y = nav_by.split_train_test(df)
nav_by.fit(train_x, train_y)
#print(nav_by.prdict([5,35,8,45,91330,4,1.00,2,0,0,0,0,0]))
y_expect = test_y
y_predict = nav_by.predict_onlist(test_x)
print(nav_by.find_accuracy(y_expect, y_predict))