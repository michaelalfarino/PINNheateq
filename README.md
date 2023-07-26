# Aplikasi Physics-Informed Neural Network (PINN) dalam Fenomena Persebaran Panas

Fenomena persebaran panas telah dimodelkan menggunakan *Physics-Informed Neural Network* (PINN). Metode ini memanfaatkan persamaan fisika yang sudah diketahui, seperti persamaan panas, ke dalam proses *training* dari *neural network*.

Proyek ini bertujuan untuk menganalisis kinerja PINN dalam menyelesaikan masalah persebaran panas dan membandingkannya dengan pendekatan metode beda-hingga serta solusi analitik ketika solusinya tersedia. Fokus ada pada mempelajari perilaku PINN untuk masalah persebaran panas sederhana, mengevaluasi akurasi dan efisiensi komputasinya.

Setelah itu, akan diperluas investigasi ke persamaan panas dua dimensi disertai dengan sumber panas lokal, yang menghadirkan tantangan karena adanya suku sumber panas. Dengan memanfaatkan fleksibilitas PINN, akan ditunjukkan kemampuan PINN untuk secara akurat menangkap perilaku rumit dari persebaran panas dengan adanya sumber panas tersebut.

- [File PDF lengkap](https://github.com/michaelalfarino/PINNheateq/blob/main/Aplikasi%20Physics-Informed%20Neural%20Network%20(PINN)%20dalam%20Fenomena%20Persebaran%20Panas.pdf)

- Code :
  * [Persamaan panas 1 dimensi](https://github.com/michaelalfarino/PINNheateq/blob/main/1d/1d.ipynb)
  * [Persamaan panas 2 dimensi](https://github.com/michaelalfarino/PINNheateq/blob/main/2d/2d.ipynb)
  * [Persamaan panas 2 dimensi dengan sumber panas di tengah domain (titik (0.5, 0.5))](https://github.com/michaelalfarino/PINNheateq/blob/main/2dhs/2dhsvar1/2dhsvar1.ipynb)
  * [Persamaan panas 2 dimensi dengan sumber panas di pinggir domain (titik (0.25, 0.25))](https://github.com/michaelalfarino/PINNheateq/blob/main/2dhs/2dhsvar2/2dhsvar2.ipynb)
