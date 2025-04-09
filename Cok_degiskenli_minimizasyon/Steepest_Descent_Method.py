import numpy as np
import sympy as sp
from sympy import Matrix, symbols, lambdify, hessian, diff
import math
import matplotlib.pyplot as plt

def steepest_descent_method():
    # Kullanıcıdan değişken sayısını al
    print("Steepest Descent (En Dik İniş) Metodu")
    print("Değişken sayısını girin (ör: 2):")
    n_vars = int(input())
    
    # Sembolik değişkenleri oluştur
    if n_vars == 2:
        x1, x2 = symbols('x1 x2')
        vars_list = [x1, x2]
    elif n_vars == 3:
        x1, x2, x3 = symbols('x1 x2 x3')
        vars_list = [x1, x2, x3]
    else:
        print(f"Değişken sayısı {n_vars} için dinamik değişkenler oluşturuluyor...")
        vars_list = symbols(', '.join([f'x{i+1}' for i in range(n_vars)]))
        if not isinstance(vars_list, list):
            vars_list = [vars_list]
    
    # Fonksiyonu al
    print(f"Fonksiyonu {', '.join([str(v) for v in vars_list])} değişkenleri cinsinden girin:")
    print("Örnek: (x1-1)**2 + (x2-1)**2")
    func_str = input()
    
    try:
        # Fonksiyonu sembolik olarak dönüştür
        f_expr = sp.sympify(func_str)
        
        # Gradyan vektörünü hesapla
        grad = Matrix([diff(f_expr, var) for var in vars_list])
        
        # Sayısal fonksiyonları oluştur
        f_func = lambdify(vars_list, f_expr, 'numpy')
        grad_func = lambdify(vars_list, grad, 'numpy')
        
        # Golden Section arama yöntemi için sembolik alfa değişkeni
        alpha = sp.Symbol('alpha')
        
        # Başlangıç noktasını al
        print(f"Başlangıç noktasını girin ({n_vars} değer, virgülle ayrılmış):")
        print(f"Örnek: 0, 0{', 0' * (n_vars-2)}")
        start_point_str = input()
        start_point = np.array([float(x.strip()) for x in start_point_str.split(',')])
        
        if len(start_point) != n_vars:
            print(f"Hata: {n_vars} değişken için {n_vars} değer girmelisiniz.")
            return
        
        # Tolerans değerleri
        print("Fonksiyon değişimi için tolerans değerini girin (varsayılan: 1e-10):")
        eps1_input = input()
        eps1 = float(eps1_input) if eps1_input else 1e-10
        
        print("Değişken değişimi için tolerans değerini girin (varsayılan: 1e-10):")
        eps2_input = input()
        eps2 = float(eps2_input) if eps2_input else 1e-10
        
        print("Gradyan normu için tolerans değerini girin (varsayılan: 1e-10):")
        eps3_input = input()
        eps3 = float(eps3_input) if eps3_input else 1e-10
        
        # Maksimum iterasyon sayısı
        print("Maksimum iterasyon sayısını girin (varsayılan: 1000):")
        max_iter_input = input()
        max_iter = int(max_iter_input) if max_iter_input else 1000
        
        # Steepest Descent algoritması
        x = start_point
        iteration = 0
        
        # Optimizasyon sürecini takip etmek için listeler
        x_history = [x.copy()]
        f_history = [f_func(*x)]
        grad_norm_history = [np.linalg.norm(grad_func(*x))]
        
        print(f"\nİterasyon {iteration}: x = {x}, f(x) = {f_func(*x)}, Gradyan Normu = {np.linalg.norm(grad_func(*x))}")
        
        updated_x = x.copy()  # İlk iterasyon için kopya
        
        while iteration < max_iter:
            # En dik iniş yönü
            pk = -grad_func(*x)
            
            # Adım boyutunu bul (Golden Section Search)
            sk = golden_section_search(f_func, x, pk, vars_list, f_expr)
            
            # Noktayı güncelle
            updated_x = x + sk * pk
            
            # Yakınsama kontrolleri
            if abs(f_func(*updated_x) - f_func(*x)) < eps1:
                print("Fonksiyon değeri yakınsaması sağlandı.")
                break
                
            if np.linalg.norm(updated_x - x) < eps2:
                print("Değişken değişimi yakınsaması sağlandı.")
                break
                
            current_grad_norm = np.linalg.norm(grad_func(*updated_x))
            if current_grad_norm < eps3:
                print("Gradyan normu yakınsaması sağlandı.")
                break
            
            # Değerleri güncelle
            x = updated_x
            iteration += 1
            
            # Geçmiş değerleri kaydet
            x_history.append(x.copy())
            f_history.append(f_func(*x))
            grad_norm_history.append(current_grad_norm)
            
            print(f"İterasyon {iteration}: x = {x}, f(x) = {f_func(*x)}, Gradyan Normu = {current_grad_norm}")
        
        if iteration == max_iter:
            print("Maksimum iterasyon sayısına ulaşıldı.")
            
        print(f"Sonuç: x = {x}, f(x) = {f_func(*x)}, Gradyan Normu = {np.linalg.norm(grad_func(*x))}")
        
        # Eğer 2 boyutlu ise grafik çiz
        if n_vars == 2:
            x_history = np.array(x_history)
            
            plt.figure(figsize=(15, 5))
            
            # Optimizasyon yolu
            plt.subplot(1, 3, 1)
            plt.plot(x_history[:, 0], x_history[:, 1], 'b-', label='Optimizasyon Yolu')
            plt.scatter(x_history[:, 0], x_history[:, 1], s=30, c='red', label='İterasyon Noktaları')
            plt.scatter(x_history[-1, 0], x_history[-1, 1], s=100, c='green', label='Sonuç')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title('Steepest Descent Optimizasyon Yolu')
            plt.legend()
            plt.grid(True)
            
            # Fonksiyon değerinin iterasyonla değişimi
            plt.subplot(1, 3, 2)
            plt.plot(range(len(f_history)), f_history, 'ro-')
            plt.xlabel('İterasyon')
            plt.ylabel('f(x)')
            plt.title('Fonksiyon Değerinin İterasyonla Değişimi')
            plt.grid(True)
            
            # Gradyan normunun iterasyonla değişimi
            plt.subplot(1, 3, 3)
            plt.plot(range(len(grad_norm_history)), grad_norm_history, 'go-')
            plt.xlabel('İterasyon')
            plt.ylabel('||∇f(x)||')
            plt.title('Gradyan Normunun İterasyonla Değişimi')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Kontur grafiği çiz (2D için)
            x1_range = np.linspace(min(x_history[:, 0]) - 1, max(x_history[:, 0]) + 1, 100)
            x2_range = np.linspace(min(x_history[:, 1]) - 1, max(x_history[:, 1]) + 1, 100)
            X1, X2 = np.meshgrid(x1_range, x2_range)
            Z = np.zeros_like(X1)
            
            for i in range(len(x1_range)):
                for j in range(len(x2_range)):
                    Z[j, i] = f_func(X1[j, i], X2[j, i])
            
            plt.figure(figsize=(10, 8))
            contour = plt.contour(X1, X2, Z, 50, cmap='viridis')
            plt.colorbar(contour, label='f(x1, x2)')
            plt.plot(x_history[:, 0], x_history[:, 1], 'r-o', linewidth=2, markersize=5, label='Optimizasyon Yolu')
            plt.scatter(x_history[-1, 0], x_history[-1, 1], s=100, c='red', marker='*', label='Sonuç')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title('Steepest Descent Optimizasyon Yolu ve Kontur Grafiği')
            plt.legend()
            plt.grid(True)
            plt.show()
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

# Golden Section Search metodu (adım boyutu optimizasyonu için)
def golden_section_search(f_func, xk, pk, vars_list, f_expr):
    # Altın oran hesaplamaları
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    
    # Aralık sınırları
    xbottom = 0
    xtop = 1
    dx = 0.00001
    
    epsilon = dx / (xtop - xbottom)
    N = round(-2.078 * math.log(epsilon))
    
    # Alfa fonksiyonu (adım boyutu optimizasyonu için)
    def f_alpha(alpha_val):
        return f_func(*(xk + alpha_val * pk))
    
    # İlk iterasyon değerleri
    x1 = xbottom + tau * (xtop - xbottom)
    f1 = f_alpha(x1)
    x2 = xtop - tau * (xtop - xbottom)
    f2 = f_alpha(x2)
    
    # Golden Section Search
    for _ in range(N):
        if f1 > f2:
            xbottom = x1
            x1 = x2
            f1 = f2
            x2 = xtop - tau * (xtop - xbottom)
            f2 = f_alpha(x2)
        else:
            xtop = x2
            x2 = x1
            f2 = f1
            x1 = xbottom + tau * (xtop - xbottom)
            f1 = f_alpha(x1)
    
    # Optimal adım boyutu
    result = 0.5 * (x1 + x2)
    return result

# Test amaçlı örnek fonksiyon
def test_example():
    print("\n=== ÖRNEK ÇÖZÜM ===")
    print("Fonksiyon: f(x1,x2) = 3 + (x1 - 1.5*x2)^2 + (x2-2)^2")
    
    # Sembolik değişkenler
    x1, x2 = symbols('x1 x2')
    vars_list = [x1, x2]
    
    # Fonksiyon tanımı
    f_expr = 3 + (x1 - 1.5*x2)**2 + (x2-2)**2
    
    # Gradyan
    grad = Matrix([diff(f_expr, var) for var in vars_list])
    
    print("\nGradyan:")
    print(f"∂f/∂x1 = {grad[0]}")
    print(f"∂f/∂x2 = {grad[1]}")
    
    # Sayısal fonksiyonlar
    f_func = lambdify(vars_list, f_expr, 'numpy')
    grad_func = lambdify(vars_list, grad, 'numpy')
    
    # Başlangıç noktası ve parametreler
    x = np.array([-4.5, -3.5])
    max_iter = 20
    eps1 = eps2 = eps3 = 1e-8
    
    print(f"\nBaşlangıç noktası: x = {x}")
    print(f"f(x) = {f_func(*x)}, Gradyan Normu = {np.linalg.norm(grad_func(*x))}")
    
    # Optimizasyon için gerekli listeler
    x_history = [x.copy()]
    f_history = [f_func(*x)]
    grad_norm_history = [np.linalg.norm(grad_func(*x))]
    
    # Steepest Descent algoritması
    iteration = 0
    updated_x = x.copy()
    
    while iteration < max_iter:
        # En dik iniş yönü
        pk = -grad_func(*x)
        
        # Adım boyutunu bul
        sk = golden_section_search(f_func, x, pk, vars_list, f_expr)
        
        # Noktayı güncelle
        updated_x = x + sk * pk
        
        # Yakınsama kontrolleri
        if abs(f_func(*updated_x) - f_func(*x)) < eps1:
            print("Fonksiyon değeri yakınsaması sağlandı.")
            break
            
        if np.linalg.norm(updated_x - x) < eps2:
            print("Değişken değişimi yakınsaması sağlandı.")
            break
            
        current_grad_norm = np.linalg.norm(grad_func(*updated_x))
        if current_grad_norm < eps3:
            print("Gradyan normu yakınsaması sağlandı.")
            break
        
        # Değerleri güncelle
        x = updated_x
        iteration += 1
        
        # Geçmiş değerleri kaydet
        x_history.append(x.copy())
        f_history.append(f_func(*x))
        grad_norm_history.append(current_grad_norm)
        
        print(f"İterasyon {iteration}: x = {x}, f(x) = {f_func(*x)}, Gradyan Normu = {current_grad_norm}")
        
        if iteration >= 5:  # İlk 5 iterasyonu göster
            print("... (kalan iterasyonlar gösterilmiyor) ...")
            break
    
    print(f"\nSonuç: x = {x}, f(x) = {f_func(*x)}, Gradyan Normu = {np.linalg.norm(grad_func(*x))}")
    print("Not: Tam sonuç için kendi fonksiyonunuzu çözün seçeneğini kullanabilirsiniz.")

if __name__ == "__main__":
    while True:
        print("\n=== Steepest Descent (En Dik İniş) Metodu ===")
        print("1. Kendi fonksiyonunuzu optimize edin")
        print("2. Örnek çözümü göster")
        print("3. Çıkış")
        
        choice = input("Seçiminiz (1/2/3): ")
        
        if choice == '1':
            steepest_descent_method()
        elif choice == '2':
            test_example()
        elif choice == '3':
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim, lütfen tekrar deneyin.")
