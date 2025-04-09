# Newton_Rapshon_Algorithm.py - Düzeltilmiş ve Dinamik Hale Getirilmiş
import numpy as np
import sympy as sp

def newton_raphson():
    # Sembolik değişken tanımlama
    x = sp.Symbol('x')
    
    # Kullanıcıdan fonksiyon girişi
    print("Newton-Raphson metodu ile kök bulma")
    print("Lütfen tek değişkenli fonksiyonu x değişkeni cinsinden girin (örn: x**2 - 4):")
    user_function = input()
    
    # Fonksiyonu sembolik olarak tanımlama
    f_expr = sp.sympify(user_function)
    
    # Türevleri hesaplama
    f1_expr = sp.diff(f_expr, x)  # Birinci türev
    f2_expr = sp.diff(f1_expr, x)  # İkinci türev
    
    # Sayısal fonksiyonları oluşturma
    f = sp.lambdify(x, f_expr, 'numpy')
    f1 = sp.lambdify(x, f1_expr, 'numpy')
    f2 = sp.lambdify(x, f2_expr, 'numpy')
    
    # Başlangıç değeri
    print("Başlangıç değerini girin:")
    x_val = float(input())
    
    # Tolerans değeri
    print("Tolerans değerini girin (varsayılan: 1e-10):")
    tol_input = input()
    tol = float(tol_input) if tol_input else 1e-10
    
    # İterasyon limiti
    print("Maksimum iterasyon sayısını girin (varsayılan: 100):")
    max_iter_input = input()
    max_iter = int(max_iter_input) if max_iter_input else 100
    
    # Newton-Raphson algoritması
    iteration = 0
    
    print(f"İterasyon {iteration}: x = {x_val}, f(x) = {f(x_val)}, f'(x) = {f1(x_val)}, f''(x) = {f2(x_val)}")
    
    while abs(f1(x_val)) > tol and iteration < max_iter:
        # Hessian matrisinin sıfıra çok yakın olması durumu
        if abs(f2(x_val)) < 1e-10:
            print("İkinci türev sıfıra çok yakın, işlem durduruluyor.")
            break
            
        dx = -f1(x_val) / f2(x_val)  # Düzelttik: işaret "-" olmalı
        x_val = x_val + dx
        
        iteration += 1
        print(f"İterasyon {iteration}: x = {x_val:.6f}, f(x) = {f(x_val):.6f}, f'(x) = {f1(x_val):.6f}, f''(x) = {f2(x_val):.6f}, dx = {dx:.6f}")
    
    if iteration == max_iter:
        print("Maksimum iterasyon sayısına ulaşıldı, yakınsama sağlanamadı.")
    else:
        print(f"Durağan nokta: x = {x_val:.10f}, fonksiyon değeri: f(x) = {f(x_val):.10f}")
        
        # Durağan noktanın türünü belirleme
        if f2(x_val) > 0:
            print("Bu nokta bir minimum noktasıdır.")
        elif f2(x_val) < 0:
            print("Bu nokta bir maksimum noktasıdır.")
        else:
            print("Bu noktanın türü belirlenemedi (f''(x) ≈ 0).")

if __name__ == "__main__":
    newton_raphson()