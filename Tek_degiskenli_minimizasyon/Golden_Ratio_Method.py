import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt

def golden_ratio_method():
    # Sembolik değişken tanımlama
    x = sp.Symbol('x')
    
    # Kullanıcıdan fonksiyon girişi
    print("Altın Oran Yöntemi ile Optimizasyon")
    print("Lütfen tek değişkenli fonksiyonu x değişkeni cinsinden girin (örn: (x-1)**2 + 3):")
    user_function = input()
    
    # Fonksiyonu sembolik olarak tanımlama
    f_expr = sp.sympify(user_function)
    
    # Türevi hesaplama
    f1_expr = sp.diff(f_expr, x)
    
    # Sayısal fonksiyonları oluşturma
    f = sp.lambdify(x, f_expr, 'numpy')
    f1 = sp.lambdify(x, f1_expr, 'numpy')
    
    # Aralık sınırlarını belirle
    print("Aralık başlangıcını girin (örn: 0):")
    x_bottom = float(input())
    
    print("Aralık sonunu girin (örn: 4):")
    x_top = float(input())
    
    # Tolerans değeri
    print("Hassasiyet değerini girin (varsayılan: 1e-6):")
    tol_input = input()
    dx = float(tol_input) if tol_input else 1e-6
    
    # Altın oran hesaplamaları
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (x_top - x_bottom)
    
    # İterasyon sayısı
    N = round(-2.078 * math.log(epsilon))
    print(f"Hesaplanan iterasyon sayısı: {N}")
    
    # İlk iterasyon için değerler
    k = 0
    x1 = x_bottom + tau * (x_top - x_bottom)
    x2 = x_top - tau * (x_top - x_bottom)
    f1_val = f(x1)
    f2_val = f(x2)
    
    # Optimizasyon sürecini göstermek için değerler
    x_values = [x1, x2]
    f_values = [f1_val, f2_val]
    iterations = [0, 0]
    
    print(f"Başlangıç: x1 = {x1:.6f}, x2 = {x2:.6f}, f(x1) = {f1_val:.6f}, f(x2) = {f2_val:.6f}")
    
    # Ana döngü
    for k in range(N):
        print(f"İterasyon {k+1}: x1 = {x1:.6f}, x2 = {x2:.6f}, f(x1) = {f1_val:.6f}, f(x2) = {f2_val:.6f}")
        
        if f1_val > f2_val:
            x_bottom = x1
            x1 = x2
            f1_val = f2_val
            x2 = x_top - tau * (x_top - x_bottom)
            f2_val = f(x2)
            
            # Değerleri kaydet
            x_values.append(x2)
            f_values.append(f2_val)
            iterations.append(k+1)
        else:
            x_top = x2
            x2 = x1
            f2_val = f1_val
            x1 = x_bottom + tau * (x_top - x_bottom)
            f1_val = f(x1)
            
            # Değerleri kaydet
            x_values.append(x1)
            f_values.append(f1_val)
            iterations.append(k+1)
    
    # Sonuç
    result = 0.5 * (x1 + x2)
    result_value = f(result)
    result_derivative = f1(result)
    
    print(f"\nSONUÇ:")
    print(f"Minimum nokta: x = {result:.10f}")
    print(f"Fonksiyon değeri: f(x) = {result_value:.10f}")
    print(f"Türev değeri: f'(x) = {result_derivative:.10f}")
    
    # Grafik çizimi için aralığı genişlet
    range_width = x_top - x_bottom
    x_plot = np.linspace(x_bottom - 0.1*range_width, x_top + 0.1*range_width, 1000)
    y_plot = [f(xi) for xi in x_plot]
    
    plt.figure(figsize=(10, 6))
    
    # Fonksiyon grafiği
    plt.subplot(2, 1, 1)
    plt.plot(x_plot, y_plot, 'b-', label='f(x)')
    plt.scatter(x_values, f_values, c='red', label='İterasyon noktaları')
    plt.scatter(result, result_value, c='green', s=100, label='Minimum')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Altın Oran Yöntemi Optimizasyonu')
    plt.legend()
    plt.grid(True)
    
    # İterasyon grafiği
    plt.subplot(2, 1, 2)
    plt.plot(iterations, f_values, 'ro-')
    plt.xlabel('İterasyon')
    plt.ylabel('f(x)')
    plt.title('Fonksiyon Değerinin İterasyonla Değişimi')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Test amaçlı örnek fonksiyon
def test_example():
    print("\n=== ÖRNEK ÇÖZÜM ===")
    print("Fonksiyon: f(x) = (x-1)^2 * (x-2) * (x-3)")
    
    x = sp.Symbol('x')
    f_expr = (x-1)**2 * (x-2) * (x-3)
    
    # Türev hesapla
    f1_expr = sp.diff(f_expr, x)
    
    print(f"Birinci türev: f'(x) = {f1_expr}")
    
    # Sayısal fonksiyonlar
    f = sp.lambdify(x, f_expr, 'numpy')
    f1 = sp.lambdify(x, f1_expr, 'numpy')
    
    # Aralık belirle
    x_bottom, x_top = 2, 3
    dx = 1e-6
    
    # Altın oran hesaplamaları
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (x_top - x_bottom)
    
    # İterasyon sayısı
    N = round(-2.078 * math.log(epsilon))
    print(f"Hesaplanan iterasyon sayısı: {N}")
    
    # İlk iterasyon için değerler
    x1 = x_bottom + tau * (x_top - x_bottom)
    x2 = x_top - tau * (x_top - x_bottom)
    f1_val = f(x1)
    f2_val = f(x2)
    
    print(f"Başlangıç: x1 = {x1:.6f}, x2 = {x2:.6f}, f(x1) = {f1_val:.6f}, f(x2) = {f2_val:.6f}")
    
    # İlk 5 iterasyonu göster
    for k in range(min(5, N)):
        print(f"İterasyon {k+1}: x1 = {x1:.6f}, x2 = {x2:.6f}, f(x1) = {f1_val:.6f}, f(x2) = {f2_val:.6f}")
        
        if f1_val > f2_val:
            x_bottom = x1
            x1 = x2
            f1_val = f2_val
            x2 = x_top - tau * (x_top - x_bottom)
            f2_val = f(x2)
        else:
            x_top = x2
            x2 = x1
            f2_val = f1_val
            x1 = x_bottom + tau * (x_top - x_bottom)
            f1_val = f(x1)
    
    # Tüm iterasyonları tamamla
    for k in range(5, N):
        if f1_val > f2_val:
            x_bottom = x1
            x1 = x2
            f1_val = f2_val
            x2 = x_top - tau * (x_top - x_bottom)
            f2_val = f(x2)
        else:
            x_top = x2
            x2 = x1
            f2_val = f1_val
            x1 = x_bottom + tau * (x_top - x_bottom)
            f1_val = f(x1)
    
    # Sonuç
    result = 0.5 * (x1 + x2)
    result_value = f(result)
    result_derivative = f1(result)
    
    print(f"\nSONUÇ:")
    print(f"Minimum nokta: x = {result:.10f}")
    print(f"Fonksiyon değeri: f(x) = {result_value:.10f}")
    print(f"Türev değeri: f'(x) = {result_derivative:.10f}")

if __name__ == "__main__":
    while True:
        print("\n=== Altın Oran Yöntemi ===")
        print("1. Kendi fonksiyonunuzu optimize edin")
        print("2. Örnek çözümü göster")
        print("3. Çıkış")
        
        choice = input("Seçiminiz (1/2/3): ")
        
        if choice == '1':
            golden_ratio_method()
        elif choice == '2':
            test_example()
        elif choice == '3':
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim, lütfen tekrar deneyin.")
