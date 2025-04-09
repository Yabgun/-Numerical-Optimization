# Çok Değişkenli Newton-Raphson Algoritması
import numpy as np
import sympy as sp
from sympy import Matrix, symbols, lambdify, hessian, diff

def newton_raphson_multivariable():
    # Kullanıcıdan değişken sayısını al
    print("Çok Değişkenli Newton-Raphson Algoritması")
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
    print("Örnek: (x1-1)**2 + (x2-1)**2 - x1*x2")
    func_str = input()
    
    try:
        # Fonksiyonu sembolik olarak dönüştür
        f_expr = sp.sympify(func_str)
        
        # Gradyan vektörünü hesapla
        grad = Matrix([diff(f_expr, var) for var in vars_list])
        
        # Hessian matrisini hesapla
        hess = hessian(f_expr, vars_list)
        
        # Sayısal fonksiyonları oluştur
        f_func = lambdify(vars_list, f_expr, 'numpy')
        grad_func = lambdify(vars_list, grad, 'numpy')
        hess_func = lambdify(vars_list, hess, 'numpy')
        
        # Başlangıç noktasını al
        print(f"Başlangıç noktasını girin ({n_vars} değer, virgülle ayrılmış):")
        print(f"Örnek: 0, 0{', 0' * (n_vars-2)}")
        start_point_str = input()
        start_point = np.array([float(x.strip()) for x in start_point_str.split(',')])
        
        if len(start_point) != n_vars:
            print(f"Hata: {n_vars} değişken için {n_vars} değer girmelisiniz.")
            return
        
        # Tolerans değeri
        print("Tolerans değerini girin (varsayılan: 1e-10):")
        tol_input = input()
        tol = float(tol_input) if tol_input else 1e-10
        
        # Maksimum iterasyon sayısı
        print("Maksimum iterasyon sayısını girin (varsayılan: 100):")
        max_iter_input = input()
        max_iter = int(max_iter_input) if max_iter_input else 100
        
        # Newton-Raphson algoritması
        x = start_point
        iteration = 0
        
        # Başlangıç değerlerini yazdır
        print(f"\nİterasyon {iteration}:")
        print(f"x = {x}")
        print(f"f(x) = {f_func(*x)}")
        grad_val = grad_func(*x)
        print(f"∇f(x) = {grad_val}")
        print(f"||∇f(x)|| = {np.linalg.norm(grad_val)}")
        
        while np.linalg.norm(grad_val) > tol and iteration < max_iter:
            iteration += 1
            
            # Hessian matrisini hesapla
            H = hess_func(*x)
            
            try:
                # Newton adımını hesapla: dx = -H^(-1) * grad
                # Hessianın determinantı sıfıra çok yakınsa veya tekil ise, küçük bir pertürbasyon ekle
                try:
                    H_inv = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    # Tekil matrise küçük bir pertürbasyon ekle
                    epsilon = 1e-6
                    H = H + epsilon * np.eye(n_vars)
                    H_inv = np.linalg.inv(H)
                    print(f"Uyarı: Hessian matrisi tekil, epsilon={epsilon} pertürbasyonu eklendi.")
                
                # Newton adımını hesapla ve noktayı güncelle
                dx = -np.dot(H_inv, grad_val.flatten())
                x = x + dx
                
                # Yeni değerleri yazdır
                grad_val = grad_func(*x)
                grad_norm = np.linalg.norm(grad_val)
                
                print(f"\nİterasyon {iteration}:")
                print(f"x = {x}")
                print(f"f(x) = {f_func(*x)}")
                print(f"∇f(x) = {grad_val}")
                print(f"||∇f(x)|| = {grad_norm}")
                print(f"dx = {dx}")
                
            except Exception as e:
                print(f"Newton adımı hesaplanırken hata oluştu: {str(e)}")
                break
        
        if iteration == max_iter:
            print("\nMaksimum iterasyon sayısına ulaşıldı. Yakınsama sağlanamadı.")
        elif np.linalg.norm(grad_val) <= tol:
            print("\nYakınsama sağlandı!")
            print(f"Durağan nokta: x* = {x}")
            print(f"Fonksiyon değeri: f(x*) = {f_func(*x)}")
            print(f"Gradyan normu: ||∇f(x*)|| = {np.linalg.norm(grad_val)}")
            
            # Hessian matrisini hesapla ve özdeğerleri bul
            H = hess_func(*x)
            print("\nHessian matrisi:")
            print(H)
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                print(f"Özdeğerler: {eigenvalues}")
                
                # Durağan noktanın türünü belirle
                if np.all(eigenvalues > 0):
                    print("Tüm özdeğerler pozitif olduğundan, bu nokta bir minimum noktasıdır.")
                elif np.all(eigenvalues < 0):
                    print("Tüm özdeğerler negatif olduğundan, bu nokta bir maksimum noktasıdır.")
                elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
                    print("Özdeğerlerde hem pozitif hem negatif değerler olduğundan, bu nokta bir eyer (semer) noktasıdır.")
                else:
                    print("Özdeğerlerde sıfır değerler olduğundan, bu noktanın türü belirlenemedi.")
            except np.linalg.LinAlgError:
                print("Hessian özdeğerleri hesaplanamadı.")
                
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

# Test amaçlı örnek fonksiyon çözümü (Rosenbrock fonksiyonu)
def test_example():
    print("\n=== ÖRNEK ÇÖZÜM ===")
    print("Verilen örnek: f(x1,x2) = (x1-1)^2 + (x2-1)^2 - x1*x2")
    
    x1, x2 = symbols('x1 x2')
    f_expr = (x1-1)**2 + (x2-1)**2 - x1*x2
    
    # Gradyan vektörünü hesapla
    grad = Matrix([diff(f_expr, var) for var in [x1, x2]])
    
    # Hessian matrisini hesapla
    hess = hessian(f_expr, [x1, x2])
    
    print("\nGradyan:")
    print(f"∂f/∂x1 = {grad[0]}")
    print(f"∂f/∂x2 = {grad[1]}")
    
    print("\nHessian matrisi:")
    print(hess)
    
    # Gradient'in sıfır olduğu noktayı bul (analitik çözüm)
    try:
        solution = sp.solve([grad[0], grad[1]], [x1, x2])
        if solution:
            print("\nDurağan noktalar (analitik çözüm):")
            for sol in solution:
                print(f"x* = {sol}")
                
                # Durağan noktada Hessian'ı değerlendir
                x1_val, x2_val = sol
                H_val = hess.subs({x1: x1_val, x2: x2_val})
                print(f"Hessian at x*: {H_val}")
                
                # Özdeğerleri sembolik olarak hesapla
                eigenvals = list(H_val.eigenvals().keys())
                print(f"Özdeğerler: {eigenvals}")
                
                # Durağan noktanın türünü belirle
                if all(val > 0 for val in eigenvals):
                    print("Bu nokta bir minimum noktasıdır.\n")
                elif all(val < 0 for val in eigenvals):
                    print("Bu nokta bir maksimum noktasıdır.\n")
                elif any(val > 0 for val in eigenvals) and any(val < 0 for val in eigenvals):
                    print("Bu nokta bir eyer (semer) noktasıdır.\n")
                else:
                    print("Bu noktanın türü belirlenemedi.\n")
    
    except Exception as e:
        print(f"Analitik çözüm hesaplanırken hata oluştu: {str(e)}")

if __name__ == "__main__":
    while True:
        print("\n=== Çok Değişkenli Newton-Raphson Algoritması ===")
        print("1. Kendi fonksiyonunuzu çözün")
        print("2. Örnek çözümü göster: f(x1,x2) = (x1-1)^2 + (x2-1)^2 - x1*x2")
        print("3. Çıkış")
        
        choice = input("Seçiminiz (1/2/3): ")
        
        if choice == '1':
            newton_raphson_multivariable()
        elif choice == '2':
            test_example()
        elif choice == '3':
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim, lütfen tekrar deneyin.") 