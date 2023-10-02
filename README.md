# ParkinsonDetectionBackEnd

Kocaeli Üniversitesi 4. Sınıf Bitirme Projesidir.

Parkinson Hastalığının ses ile tespiti yapılmaktadır. Bir Android uygulamasıdır. Kotlin ile yazılmıştır. FrontEnd kısmı Python ile yazılmıştır (https://github.com/emirayranci18/ParkinsonDetectionFrontEnd). Makine öğrenmesi ve derin öğrenme kullanılmıştır. Dataset olarak https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set kullanılmıştır. Arkada bir webservice ile tüm işlemler yürütülmektedir.

Telefon üzerinden kaydedilen ses anlık olarak eğer sunucu aktif ise sunucuya gönderilir. Sunucuda test işlemi yapılır ve sonuç telefona webservice üzerinden yeniden gönderilir ve kullanıcı telefon üzerinde hastalık tahmin sonucunu görüntüler.
