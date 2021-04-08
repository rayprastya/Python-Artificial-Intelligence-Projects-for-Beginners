import unittest


class TestApp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_02_rolly_113040087(self):
        from Chapter01.rolly113040087 import preparation,training,testing
        dataset='Chapter01/dataset/student-por.csv'
        d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass= preparation(dataset)
        t = training(d_train_att,d_train_pass)
        hasiltestingsemua = testing(t,d_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    def test_03_angga_1184047(self):
        from Chapter01.angga1184047 import preprocessing, training, predict

        while True:
            data = preprocessing()

            # training data
            data_train = data['training']
            data_training = data_train['data_training']
            data_training_label = data_train['data_training_label']

            # testing data
            data_train = data['testing']
            data_testing = data_train['data_testing']
            data_testing_label = data_train['data_testing_label']

            # training
            t = training(data_training, data_training_label)

            # predict
            prediction = predict(t, data_testing)

            print(data_testing_label.values[0])
            print(prediction[0])
            if data_testing_label.values[0] == prediction[0]:
                print("hasil sama")
                break
            else:
                print("hasil beda")
        self.assertEqual(data_testing_label.values[0], prediction[0])

    def test_02_DindaMajesty_1184011(self):
        from Chapter01.DindaMajesty1184011 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/company_data.csv'
        # testing function preparation
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = preparation(datasetpath)
        #testing function training
        t = training(d_train_att, d_train_pass)
        #testing function testing
        hasiltestingsemua = testing(t, d_test_att)
        #hasil
        print('\n hasil testing dinda : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_02_DyningAida_1184030(self):
        from Chapter01.DyningAida1184030 import preparation, training, testing
        
        #path ke dataset
        datasetpath = 'Chapter01/dataset/online_shoppers_intention.csv'
        
        # testing function preparation
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = preparation(datasetpath)
        
        #testing function training
        t = training(d_train_att, d_train_pass)
        
        #testing function testing
        hasiltestingsemua = testing(t, d_test_att)
        
        #hasil
        print('\n hasil testing Batris : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    def test_02_idam_1184063(self):
        from Chapter01.idam1184063 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        dfs_train_att = train.pop(0)
        dfs_train_win = train.pop(0)
        #test data
        test = data.pop(0)
        dfs_test_att = test.pop(0)
        dfs_test_win = test.pop(0)
        #training
        t = training(dfs_train_att, dfs_train_win)
        #predict
        result = testing(t,dfs_test_att)
        print("result : ")
        print(result)
        self.assertLessEqual(result[0], 1)
        
    def test_02_alifiaZahra_1184051(self):
        from Chapter01.alifiaZahra1184051 import preparation, training, testing
        dataset ='Chapter01/dataset/bank-additional-full.csv'
        d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass= preparation(dataset)
        t = training(d_train_att,d_train_pass)
        allresult = testing(t,d_test_att)
        print('\n hasil testing : ')
        print(allresult)
        oneresult = allresult[0]
        self.assertLessEqual(oneresult, 1)

    def test_02_AhmadAgung_1184015(self):
        from Chapter01.AhmadAgung1184015 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        vg_train_att = train.pop(0)
        vg_train_gbs = train.pop(0)
        #test data
        test = data.pop(0)
        vg_test_att = test.pop(0)
        vg_test_gbs = test.pop(0)
        #training
        t = training(vg_train_att, vg_train_gbs)
        #predict
        result = testing(t,vg_test_att)
        print("result : ")
        print(result)
        self.assertLessEqual(result[0], 1)
        

    def test_02_FarisMuhammadIhsan_1184099(self):
        from Chapter01.FarisIhsan1184099 import preparation, train, test
        
        #path ke dataset
        dataset = 'Chapter01/dataset/stroke.csv'
        
        # testing function preparation
        d_train_att, d_train_stroke, d_test_att, d_test_stroke, d_att, d_stroke = preparation(dataset)
        
        #testing function training
        t = train(d_train_att, d_train_stroke)
        
        #testing function testing
        hasiltestingsemua = test(t, d_test_att)
        
        #hasil
        print('\n hasil test Faris :')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    
    def test_02_mwahyu_1184059(self):
        from Chapter01.mwahyu1184059 import preparation,training,testing
         #data
        data = preparation()
        #train data
        train = data.pop(0)
        dta_train_att = train.pop(0)
        dta_train_outcome = train.pop(0)
        #test data
        test = data.pop(0)
        dta_test_att = test.pop(0)
        dta_test_outcome = test.pop(0)
        #training
        t = training(dta_train_att, dta_train_outcome)
        #predict
        result = testing(t,dta_test_att)
        print("result : ")
        print(result)
        self.assertLessEqual(result[0], 1)

    def test_02_rayhany_1184007(self):
        from Chapter01.rayhanyuda1184007 import preparation,training,testing
        #data
        dt = preparation()
        #train data
        train = dt.pop(0)
        dfrs_train_atribut = train.pop(0)
        dfrs_train_sick = train.pop(0)
        #test data
        test = dt.pop(0)
        dfrs_test_atribut = test.pop(0)
        dfrs_test_sick = test.pop(0)
        #training
        r = training(dfrs_train_atribut, dfrs_train_sick)
        #predict
        output = testing(r,dfrs_test_atribut)
        print("output test: ")
        print(output)
        self.assertLessEqual(output[0], 1)
    
    def test_02_rayhanprastya_1184069(self):
        from Chapter01.rayhanprastya1184069 import preparation,training, testing
        datasetpath = 'Chapter01/dataset/spambase.csv'
        data = preparation(datasetpath)
        # data train
        dat_train = data.pop(0)
        dat_train_atr = dat_train.pop(0)
        dat_train_cls = dat_train.pop(0)
        # data test
        dat_test = data.pop(0)
        dat_test_atr = dat_test.pop(0)
        dat_test_cls = dat_test.pop(0)
        # training data
        trainingg = training(dat_test_atr,dat_test_cls)
        # data predict
        hasil = testing(trainingg,dat_test_atr)
        print("hasil testing spam : ")
        print(hasil)
        self.assertLessEqual(hasil[0], 1)
    
    def test_02_adityar_1184021(self):
        from Chapter01.adityar1184021 import preparation, training, testing
        #datasetpath = 'Chapter01/dataset/kuli_ah_daring.csv'
        data = preparation()

        train = data.pop(0)
        d_train_att = train.pop(0)
        d_train_pass = train.pop(0)

        test = data.pop(0)
        d_test_att = test.pop(0)
        d_test_pass = test.pop(0)

        t = training(d_train_att, d_train_pass)

        result = testing(t,d_test_att)
        print("Maka yang di approve adalah : ")
        print(result)
        self.assertGreaterEqual(result[0],0)

    def test_03_dindamajesty_1184011(self):
        from Chapter02.DindaMajesty1184011 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/mushrooms.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing dinda : ')
        print(hasiltesting)
        print('Score:', clf.score(df_test_att, df_test_label))
         
    def test_03_DyningAida_1184030(self):
        from Chapter02.DyningAida1184030 import preparation, training, testing
        dataset = 'Chapter01/dataset/nursery.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(dataset)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasil = testing(clf, df_test_att.head())
        # hasil testing
        print('\nhasil testing Batris :', hasil)
        print('Score:', clf.score(df_test_att, df_test_label))
 
    def test_03_AhmadAgung_1184015(self):
        from Chapter02.AhmadAgung_1184015 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/connect-4.txt'
        f_train_att, f_train_label, f_test_att, f_test_label, f_att, f_label = preparation(datasetpath)
        #testing dari fungsi traning
        clf = training(f_train_att, f_train_label)
        #testing dari fungsi testing
        hasiltesting = testing(clf, f_test_att.head())

        #hasil testing yang dilakukan        
        print(' testing : ')
        print(hasiltesting)
        print('Hasil draw(0) lose(1) win(2)',clf.score(f_test_att, f_test_label))

    def test_03_IdamFadilah_1184063(self):
        from Chapter02.IdamFadilah1184063 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : "+ str(t.score(testAttr, testVar)))

        
    def test_04_Nurhanifah_1184086(self):
        from Chapter02.Nurhanifah1184086 import preparation, training, testing
        dataset = 'Chapter01/dataset/Callt.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(dataset)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasil = testing(clf, df_test_att.head())
        # hasil testing
        print('\nhasil testing hanifah :', hasil)
        print('Score:', clf.score(df_test_att, df_test_label))
        self.assertLessEqual(hasil[1], 1)
    
    def test_03_alifiaZahra_1184051(self):
        from Chapter02.alifiaZahra1184051 import preparation, training, testing
        datasetpath ='Chapter01/dataset/poker-hand2.txt'
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        clf = training(df_train_att, df_train_label)
        allresult = testing(clf, df_test_att.head())
        print('\n hasil testing : ', allresult)
        print('Score:', clf.score(df_test_att, df_test_label))
        oneresult = allresult[0]
        self.assertLessEqual(oneresult,1)
        
    def test_03_FarisMuhammadIhsan_1184099(self):
        from Chapter02.FarisMuhammadIhsan1184099 import preparation, training, testing
        datasetpath = 'Chapter01/dataset/jamurclassf.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasil = testing(clf, df_test_att.head())
        # hasil testing
        print('\nhasil testing Faris :', hasil)
        print('Score:', clf.score(df_test_att, df_test_label))
        self.assertLessEqual(hasil[0], 1)

    def test_03_Rayhanyl_1184007(self):
        from Chapter02.Rayhanyl1184007 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        train_Attribut = train.pop(0)
        train_Varr = train.pop(0)

        test_Attribut = test.pop(0)
        test_Varr = test.pop(0)

        t = training(train_Attribut, train_Varr)

        hasil = testing(t, test_Attribut)
        print('hasil : ')
        print(hasil)
        print("score : ", t.score(test_Attribut, test_Varr))
        self.assertLessEqual(hasil[0], 1)
    
    def test_03_EtikaKhusnulLaeli_1184065(self):
        from Chapter02.Etika1184065 import preparation, training, testing

        dataset = 'Chapter01/dataset/datatraining.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(dataset)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing etika : ')
        print(hasiltesting)
        print('Score:', clf.score(df_test_att, df_test_label))
        self.assertLessEqual(hasiltesting[0],1)
    
    def test_03_Anurutari_1184039(self):
        from Chapter02.Anurutari1184039 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trn_Atribut = train.pop(0)
        trn_Varr = train.pop(0)

        test_Atribut = test.pop(0)
        test_Varr = test.pop(0)

        t = training(trn_Atribut, trn_Varr)

        hasiltest = testing(t, test_Atribut)
        print('hasil : ')
        print(hasiltest)
        print("score : ", t.score(test_Atribut, test_Varr))
        self.assertLessEqual(hasiltest[0], 1)

    def test_03_adityar_1184021(self):
        from Chapter02.adityar_1184021 import preparation, training, testing
        data = preparation()
        train = data.pop(0)
        test = data.pop(0)
        trainAttr = train.pop(0)
        trainLabel = train.pop(0)
        testAttr = test.pop(0)
        testLabel = test.pop(0)
        t = training(trainAttr, trainLabel)
        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("Hasilnya Adalah : "+ str(t.score(testAttr, testLabel)))

    def test_03_SitiNPujaKesuma_1184004(self):
        from Chapter02.SitiNPujaKesuma1184004 import preparation, training, testing
        info = preparation()

        train = info.pop(0)
        test = info.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ",t.score(testAttr, testVar))
        self.assertLessEqual(result[0],1)
    
    def test_03_rayhanprastya_1184069(self):
        from Chapter02.rayhanprastya1184069 import preparation, training, testing
        dataset = 'Chapter01/dataset/cardio.txt'
        data = preparation(dataset)

        train = data.pop(0)
        test = data.pop(0)

        trainatr = train.pop(0)
        trainvar = train.pop(0)

        testatr = test.pop(0)
        testvar = test.pop(0)

        a = training(trainatr, trainvar)
        hasil = testing(a, testatr)

        print('hasil test : ')
        print(hasil)
        print("score testing : ", a.score(testatr, testvar))
        self.assertLessEqual(hasil[0],1)
    
    def test_04_idamFadilah_1184063(self):
        from Chapter03.idamFadilah1184063 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ",t.score(testAttr, testVar))

    def test_04_muhammadwahyu_1184059(self):
        from Chapter03.muhammadwahyu1184059 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ",t.score(testAttr, testVar))
        self.assertLessEqual(result[0], 1)

    def test_04_Rayhanyl_1184007(self):
        from Chapter03.rayhanyuda1184007 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ",t.score(testAttr, testVar))
        self.assertLessEqual(result[0], 1)
    
    def test_04_DyningAida_1184030(self):
        from Chapter03.DyningAida1184030 import preparation, training, testing
        dataset = 'Chapter01/dataset/books-rating.csv'
        d_train_att, d_train_label, d_test_att, d_test_label = preparation(dataset)
        clf = training(d_train_att, d_train_label)
        # testing function testing
        hasil = testing(clf, d_test_att)
        # hasil testing
        print('\nhasil testing Batris :', hasil)
        print('Score:', clf.score(d_test_att, d_test_label))
        self.assertLessEqual(hasil[0],2)
        
    def test_04_AhmadAgung_1184015(self):
        from Chapter03.AhmadAgung_1184015 import preparation, training, testing
        datasetmovie = 'Chapter01/dataset/movie_ratings.csv'
        mv_train_att, mv_train_label, mv_test_att, mv_test_label = preparation(datasetmovie)
        clf = training(mv_train_att, mv_train_label)
        
        hasilvote = testing(clf, mv_test_att)
        
        print(' 0 = rating film < 7')
        print(' 1 = rating film > 7')
        print(hasilvote)
        print(' IMDB:', clf.score(mv_test_att, mv_test_label))
        self.assertLessEqual(hasilvote[0],1)

    def test_04_Anurutari_1184039(self):
        from Chapter03.Anurutari1184039 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ",t.score(testAttr, testVar))
        self.assertLessEqual(result[0], 1)

    def test_04_DindaMajesty_1184011(self):
        from Chapter03.DindaMajesty1184011 import preparation, training, testing
        dataset = 'Chapter01/dataset/ramen-ratings.csv'
        d_train_att, d_train_label, d_test_att, d_test_label = preparation(dataset)
        clf = training(d_train_att, d_train_label)
        # testing function testing
        hasil = testing(clf, d_test_att)
        # hasil testing
        print('3: Very Recommended, 2: Recommended, 1: Less Recommended, 0: Not Recommended')
        print('\nTesting Dinda :', hasil)
        print('Score:', clf.score(d_test_att, d_test_label))
        self.assertLessEqual(hasil[0],3)

    def test_04_SitiNPujaKesuma_1184004(self):
        from Chapter03.SitiNPujaKesuma1184004 import preparation, training, testing
        baca = preparation()

        train = baca.pop(0)
        test = baca.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ",t.score(testAttr, testVar))
        self.assertLessEqual(result[0], 1)
        
    def test_05_Nurhanifah_1184086(self):
        from Chapter03.Nurhanifah1184086 import preparation, training, testing
        dataset = 'Chapter01/dataset/Food_Reviews.csv'
        d_train_att, d_train_label, d_test_att, d_test_label = preparation(dataset)
        clf = training(d_train_att, d_train_label)
        # testing function testing
        hasil = testing(clf, d_test_att)
        # hasil testing
        print('\nhasil testing hanifah :', hasil)
        print('Score:', clf.score(d_test_att, d_test_label))
        self.assertLessEqual(hasil[0],1)
    
    def test_04_EtikaKhusnulLaeli_1184065(self):
        from Chapter03.EtikaKhusnul1184065 import preparation, training, testing
        dataset = 'Chapter01/dataset/OnlineNewsPopularity.csv'
        d_train_att, d_train_label, d_test_att, d_test_label = preparation(dataset)
        clf = training(d_train_att, d_train_label)
        # testing function testing
        hasil = testing(clf, d_test_att)
        # hasil testing
        print('\nhasil testing Etika :', hasil)
        print('Score:', clf.score(d_test_att, d_test_label))
        self.assertLessEqual(hasil[0],1)

    def test_04_adityar_1184021(self):
        from Chapter03.adityar_1184021 import preparation, training, testing
        data = preparation()
        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('Data testingnya : ')
        print(result)
        print("Hasilnya : ",t.score(testAttr, testVar))
