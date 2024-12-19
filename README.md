Proje Hakkında (For Turkish)

Bu projede, "database" klasörü içerisinde bulunan "original images" ve "tagged images" klasörlerindeki 10 adet örnek görüntü kullanılarak Mask-RCNN tabanlı bir yapay zeka modeli oluşturuldu. Bu model, dişlerin ve kanalların konumunu tespit ederek koordinat bilgilerini ve doğruluk oranlarını sağlamaktadır.

Kullanılan Kodlar

Model Eğitimi: kanal_ve_dis_egitim.py dosyası, orijinal ve etiketlenmiş görüntüler kullanılarak Mask-RCNN modelini eğitmek için kullanılmıştır.

Model Kullanımı: kanal_ve_dis_tespiti.py dosyası, eğitilen modeli kullanarak girdi olarak verilen görüntülerde diş ve kanal tespit işlemini gerçekleştirir. Bu tespit sonucunda, tespit edilen dişlerin ve kanalların koordinatları ile doğruluk oranları görüntülenir.

Örnek Tahmin Görüntüleri

Tahmin edilen görüntüler, "database" klasörü içindeki "predicted images" klasöründe bulunmaktadır. Bu klasörde, modelin girdi görüntülerine yaptığı tespitlerin çıktıları yer almaktadır.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
About the Project (For English)

In this project, a Mask-RCNN-based artificial intelligence model was created using 10 sample images from the "original images" and "tagged images" folders located in the "database" directory. This model identifies the positions of teeth and canals, providing their coordinate information and accuracy rates.

Used Scripts

Model Training: The kanal_ve_dis_egitim.py script is used to train the Mask-RCNN model using the original and tagged images.

Model Usage: The kanal_ve_dis_tespiti.py script applies the trained model to detect teeth and canals in the input images. As a result, the coordinates and accuracy rates of the detected teeth and canals are displayed.

Sample Prediction Images

The predicted images can be found in the "predicted images" folder within the "database" directory. This folder contains the output of the model's detections on the input images.
