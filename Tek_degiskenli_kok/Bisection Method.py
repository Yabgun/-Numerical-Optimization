# Bisection Method.py - Düzeltilmiş ve Dinamik Hale Getirilmiş
import numpy as np
import sympy as sp

def bisection_method():
    # Sembolik değişken tanımlama
    x = sp.Symbol('x')
    
    # Kullanıcıdan fonksiyon girişi
    print("Bisection (İkiye Bölme) metodu ile kök bulma")
    print("Lütfen tek değişkenli fonksiyonu x değişkeni cinsinden girin (örn: x**2 - 4):")
    user_function = input()
    
    # Fonksiyonu sembolik olarak tanımlama
    f_expr = sp.sympify(user_function)
    
    # Türevi hesaplama (durağan nokta bulma için)
    f1_expr = sp.diff(f_expr, x)
    
    # Sayısal fonksiyonları oluşturma
    f = sp.lambdify(x, f_expr, 'numpy')
    f1 = sp.lambdify(x, f1_expr, 'numpy')
    
    # Aralık sınırlarını belirle
    print("Aralık başlangıcını girin (örn: -5):")
    x1 = float(input())
    
    print("Aralık sonunu girin (örn: 5):")
    x2 = float(input())
    
    # Tolerans değeri
    print("Tolerans değerini girin (varsayılan: 1e-9):")
    tol_input = input()
    tol = float(tol_input) if tol_input else 1e-9
    
    print("Kök arıyor musunuz (f(x)=0) yoksa durağan nokta mı (f'(x)=0)? (kök/durağan):")
    search_type = input().lower()
    
    # Hangi fonksiyonun kökünü arayacağımızı belirle
    if search_type == "durağan" or search_type == "duragan":
        target_f = f1
        print(f"Durağan nokta arama: f'({x1}) = {f1(x1)}, f'({x2}) = {f1(x2)}")
    else:
        target_f = f
        print(f"Kök arama: f({x1}) = {f(x1)}, f({x2}) = {f(x2)}")
    
    # Aralıkta işaret değişimi kontrol et
    if target_f(x1) * target_f(x2) >= 0:
        print(f"Bu aralıkta ({x1}, {x2}) kök olmayabilir, işaret değişimi yok!")
        return
    
    iteration = 0
    
    while abs(x2 - x1) > tol:
        # Orta noktayı hesapla
        xk = (x1 + x2) / 2
        iteration += 1
        
        # Orta noktadaki değeri hesapla
        fk = target_f(xk)
        
        if abs(fk) < tol:
            break
        
        # Aralığı güncelle
        if target_f(x1) * fk < 0:
            x2 = xk
        else:
            x1 = xk
            
        print(f"İterasyon {iteration}: x = {xk:.10f}, {'f(x)' if search_type != 'durağan' else 'f\'(x)'} = {fk:.10f}")
    
    # Sonucu göster
    if search_type == "durağan" or search_type == "duragan":
        print(f"Durağan nokta: x = {xk:.10f}, f(x) = {f(xk):.10f}, f'(x) = {f1(xk):.10f}")
        
        # İkinci türevi hesapla
        f2_expr = sp.diff(f1_expr, x)
        f2 = sp.lambdify(x, f2_expr, 'numpy')
        f2_val = f2(xk)
        
        # Durağan noktanın türünü belirle
        if f2_val > 0:
            print("Bu nokta bir minimum noktasıdır.")
        elif f2_val < 0:
            print("Bu nokta bir maksimum noktasıdır.")
        else:
            print("Bu noktanın türü belirlenemedi (f''(x) ≈ 0).")
    else:
        print(f"Kök: x = {xk:.10f}, f(x) = {f(xk):.10f}")

# Test amaçlı örnek fonksiyon
def test_example():
    print("\n=== ÖRNEK ÇÖZÜM ===")
    print("Fonksiyon: f(x) = x^3 - 6x^2 + 11x - 6")
    
    x = sp.Symbol('x')
    f_expr = x**3 - 6*x**2 + 11*x - 6
    
    # Türev hesapla
    f1_expr = sp.diff(f_expr, x)
    
    print(f"Birinci türev: f'(x) = {f1_expr}")
    print(f"Kökler: x=1, x=2, x=3 (anlitik çözüm)\n")
    
    # Sayısal fonksiyonlar
    f = sp.lambdify(x, f_expr, 'numpy')
    f1 = sp.lambdify(x, f1_expr, 'numpy')
    
    # Kök bulma örneği
    x1, x2 = 0.5, 1.5
    print(f"Kök arama: [x1, x2] = [{x1}, {x2}]")
    print(f"f(x1) = {f(x1):.6f}, f(x2) = {f(x2):.6f}")
    
    if f(x1) * f(x2) < 0:
        print("Bu aralıkta kök var.\n")
        
        # Bisection yöntemi
        iteration = 0
        tol = 1e-9
        
        while abs(x2 - x1) > tol:
            xk = (x1 + x2) / 2
            iteration += 1
            
            if abs(f(xk)) < tol:
                break
                
            if f(x1) * f(xk) < 0:
                x2 = xk
            else:
                x1 = xk
                
            print(f"İterasyon {iteration}: x = {xk:.10f}, f(x) = {f(xk):.10f}")
        
        print(f"Bulunan kök: x = {xk:.10f}, f(x) = {f(xk):.10f}")
    else:
        print("Bu aralıkta kök yok.")

if __name__ == "__main__":
    while True:
        print("\n=== Bisection Method (İkiye Bölme Yöntemi) ===")
        print("1. Kendi fonksiyonunuzu çözün")
        print("2. Örnek çözümü göster")
        print("3. Çıkış")
        
        choice = input("Seçiminiz (1/2/3): ")
        
        if choice == '1':
            bisection_method()
        elif choice == '2':
            test_example()
        elif choice == '3':
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim, lütfen tekrar deneyin.")