# Aplikasi Physics-Informed Neural Network (PINN) dalam Fenomena Persebaran Panas

![Python](https://img.shields.io/badge/Python-black?logo=python)
![DeepXDE](https://img.shields.io/badge/DeepXDE-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-black?logo=tensorflow)
![NumPy](https://img.shields.io/badge/NumPy-black?logo=numpy)
![Matplotlib](https://img.shields.io/badge/matplotlib-black?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMjgiIGhlaWdodD0iMTI4IiBzdHJva2U9IiM3NzciIGZpbGwtb3BhY2l0eT0iLjgiPgo8cGF0aCBmaWxsPSIjRkZGIiBkPSJtNjMsMWE2Myw2MyAwIDEsMCAyLDB6bTAsMTRhNDksNDkgMCAxLDAgMiwwem0wLDE0YTM1LDM1IDAgMSwwCjIsMHptMCwxNGEyMSwyMSAwIDEsMCAyLDB6bTAsMTRhNyw3IDAgMSwwIDIsMHptNjQsN0gxbTEwOC00NS05MCw5MG05MCwwLTkwLTkwbTQ1LTE4djEyNiIvPgo8cGF0aCBmaWxsPSIjRjYwIiBkPSJtNTAsOC0yMCwxMCA2OCw5MiAxMC0xMEw2NCw2NHoiLz4KPHBhdGggZmlsbD0iI0ZDMCIgZD0ibTE3LDUwdjI4TDY0LDY0eiIvPgo8cGF0aCBmaWxsPSIjN0Y3IiBkPSJtNjQsNjQgNiwzNUg1OHoiLz4KPHBhdGggZmlsbD0iI0NGMyIgZD0ibTY0LDY0IDEzLTQwIDksNXoiLz4KPHBhdGggZmlsbD0iIzA0RiIgZD0ibTY0LDY0IDE0LTYgMSw0emwtMjYsMTMgMyw0eiIvPgo8L3N2Zz4=)

Fenomena persebaran panas telah dimodelkan menggunakan *Physics-Informed Neural Network* (PINN). Metode ini memanfaatkan persamaan fisika yang sudah diketahui, seperti persamaan panas, ke dalam proses *training* dari *neural network*.

Proyek ini bertujuan untuk menganalisis kinerja PINN dalam menyelesaikan masalah persebaran panas dan membandingkannya dengan pendekatan metode beda hingga serta solusi analitik ketika solusinya tersedia. Fokus ada pada mempelajari perilaku PINN untuk masalah persebaran panas sederhana, mengevaluasi akurasi dan efisiensi komputasinya.

Setelah itu, akan diperluas investigasi ke persamaan panas dua dimensi disertai dengan sumber panas lokal, yang menghadirkan tantangan karena adanya suku sumber panas. Dengan memanfaatkan fleksibilitas PINN, akan ditunjukkan kemampuan PINN untuk secara akurat menangkap perilaku rumit dari persebaran panas dengan adanya sumber panas tersebut.

Kesimpulan yang diperoleh adalah PINN menunjukkan *mean-squared error* (MSE) yang lebih kecil dibandingkan metode beda hingga, membuatnya menjadi alternatif yang menjanjikan untuk menyelesaikan persamaan panas. Keuntungan utama PINN adalah kemampuannya menghasilkan solusi yang kontinu, penting untuk masalah-masalah lanjutan yang memerlukan solusi *smooth* dan *differentiable*.

- [File PDF lengkap](https://github.com/michaelalfarino/PINNheateq/blob/main/Aplikasi%20Physics-Informed%20Neural%20Network%20(PINN)%20dalam%20Fenomena%20Persebaran%20Panas.pdf)

- Code :
  * [Persamaan panas 1 dimensi](https://github.com/michaelalfarino/PINNheateq/blob/main/1d/1d.ipynb)
  * [Persamaan panas 2 dimensi](https://github.com/michaelalfarino/PINNheateq/blob/main/2d/2d.ipynb)
  * [Persamaan panas 2 dimensi dengan sumber panas di tengah domain (titik (0.5, 0.5))](https://github.com/michaelalfarino/PINNheateq/blob/main/2dhs/2dhsvar1/2dhsvar1.ipynb)
  * [Persamaan panas 2 dimensi dengan sumber panas di pinggir domain (titik (0.25, 0.25))](https://github.com/michaelalfarino/PINNheateq/blob/main/2dhs/2dhsvar2/2dhsvar2.ipynb)
