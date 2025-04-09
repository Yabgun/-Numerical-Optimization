import numpy as np
import sympy as sp
from sympy import Matrix, symbols, lambdify, hessian, diff

def determine_stationary_point_type():
    # Kullanıcıdan değişken sayısını al
    print("Durağan Nokta Türü Belirleme")
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
        
        # Hessian matrisini hesapla
        hess = hessian(f_expr, vars_list)
        
        # Sayısal fonksiyonları oluştur
        f_func = lambdify(vars_list, f_expr, 'numpy')
        grad_func = lambdify(vars_list, grad, 'numpy')
        hess_func = lambdify(vars_list, hess, 'numpy')
        
        print("\nGradyan:")
        for i, g in enumerate(grad):
            print(f"∂f/∂x{i+1} = {g}")
        
        print("\nHessian matrisi:")
        print(hess)
        
        # Durağan noktanın analitik çözümünü bulmaya çalış
        print("\nDurağan nokta, gradyanın sıfır olduğu noktadır (∇f(x) = 0)")
        print("Analitik çözüm arıyor...")
        
        try:
            # Durağan noktaları analitik olarak bul
            solutions = sp.solve([g for g in grad], vars_list)
            
            if not solutions:
                print("Analitik çözüm bulunamadı.")
            else:
                print(f"Bulunan durağan noktalar: {solutions}")
                
                # Her bir durağan nokta için Hessian matrisini değerlendir
                for i, sol in enumerate(solutions):
                    if isinstance(sol, tuple):
                        point = sol
                    else:
                        point = tuple(sol[var] for var in vars_list)
                    
                    print(f"\nDurağan Nokta {i+1}: {point}")
                    
                    # Değerlendirme noktası
                    subs_dict = {var: val for var, val in zip(vars_list, point)}
                    
                    # Hessian matrisini değerlendir
                    H_val = hess.subs(subs_dict)
                    print(f"Bu noktada Hessian matrisi:")
                    print(H_val)
                    
                    # Özdeğerleri hesapla
                    eigenvals = list(H_val.eigenvals().keys())
                    print(f"Özdeğerler: {eigenvals}")
                    
                    # Durağan noktanın türünü belirle
                    if all(val > 0 for val in eigenvals):
                        print("Bu nokta bir minimum noktasıdır.")
                    elif all(val < 0 for val in eigenvals):
                        print("Bu nokta bir maksimum noktasıdır.")
                    elif any(val > 0 for val in eigenvals) and any(val < 0 for val in eigenvals):
                        print("Bu nokta bir eyer (semer) noktasıdır.")
                    else:
                        print("Bu noktanın türü belirlenemedi (sıfır özdeğer var).")
        
        except Exception as e:
            print(f"Analitik çözüm hesaplanırken hata oluştu: {str(e)}")
        
        # Manuel olarak bir noktada değerlendirme yapma seçeneği
        print("\nBir noktada durağan nokta analizi yapmak ister misiniz? (e/h)")
        choice = input().lower()
        
        if choice == 'e' or choice == 'evet':
            print(f"Analiz edilecek noktayı girin ({n_vars} değer, virgülle ayrılmış):")
            print(f"Örnek: 1, 1{', 1' * (n_vars-2)}")
            point_str = input()
            point = np.array([float(x.strip()) for x in point_str.split(',')])
            
            if len(point) != n_vars:
                print(f"Hata: {n_vars} değişken için {n_vars} değer girmelisiniz.")
                return
            
            # Fonksiyon, gradyan ve Hessian değerlerini hesapla
            f_val = f_func(*point)
            grad_val = grad_func(*point)
            hess_val = hess_func(*point)
            
            print(f"\nNoktada fonksiyon değeri: f({point}) = {f_val}")
            print(f"Gradyan: ∇f({point}) = {grad_val}")
            print(f"Gradyan normu: ||∇f({point})|| = {np.linalg.norm(grad_val)}")
            
            print("\nHessian matrisi:")
            print(hess_val)
            
            # Özdeğerleri hesapla
            try:
                eigenvals, eigenvecs = np.linalg.eigh(hess_val)
                print(f"Özdeğerler: {eigenvals}")
                print("Özvektörler:")
                for i, vec in enumerate(eigenvecs.T):
                    print(f"v{i+1} = {vec}")
                
                # Durağan noktanın türünü belirle
                if np.all(eigenvals > 0):
                    print("\nBu nokta bir minimum noktasıdır (tüm özdeğerler pozitif).")
                elif np.all(eigenvals < 0):
                    print("\nBu nokta bir maksimum noktasıdır (tüm özdeğerler negatif).")
                elif np.any(eigenvals > 0) and np.any(eigenvals < 0):
                    print("\nBu nokta bir eyer (semer) noktasıdır (hem pozitif hem negatif özdeğerler var).")
                else:
                    print("\nBu noktanın türü belirlenemedi (sıfır özdeğer var).")
                
                # Gradyan değeri sıfıra yakın değilse, uyarı ver
                if np.linalg.norm(grad_val) > 1e-6:
                    print(f"\nUyarı: Bu nokta bir durağan nokta olmayabilir, çünkü gradyan normunun değeri yüksek: {np.linalg.norm(grad_val)}")
                    
            except np.linalg.LinAlgError:
                print("Hessian özdeğerleri hesaplanamadı.")
                
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

# Test amaçlı örnek fonksiyon
def test_example():
    print("\n=== ÖRNEK ÇÖZÜM ===")
    print("Fonksiyon: f(x1,x2) = (x1-1)^2 + (x2-1)^2 - x1*x2")
    
    # Sembolik değişkenler
    x1, x2 = symbols('x1 x2')
    vars_list = [x1, x2]
    
    # Fonksiyon tanımı
    f_expr = (x1-1)**2 + (x2-1)**2 - x1*x2
    
    # Gradyan
    grad = Matrix([diff(f_expr, var) for var in vars_list])
    
    print("\nGradyan:")
    print(f"∂f/∂x1 = {grad[0]}")
    print(f"∂f/∂x2 = {grad[1]}")
    
    # Hessian matrisi
    hess = hessian(f_expr, vars_list)
    
    print("\nHessian matrisi:")
    print(hess)
    
    print("\nDurağan nokta için denklem sistemini çözüyoruz:")
    print("2*(x1-1) - x2 = 0")
    print("2*(x2-1) - x1 = 0")
    
    # Durağan noktaları analitik olarak bul
    solutions = sp.solve([g for g in grad], vars_list)
    
    if solutions:
        print(f"\nBulunan durağan nokta: {solutions}")
        
        # Durağan noktada Hessian'ı değerlendir
        if isinstance(solutions, list):
            # Liste şeklinde çözümler varsa (multiple solutions)
            sol = solutions[0]
            if isinstance(sol, dict):
                x1_val = sol[x1]
                x2_val = sol[x2]
            else:
                x1_val, x2_val = sol
        elif isinstance(solutions, dict):
            # Sözlük şeklinde çözüm varsa (single solution)
            x1_val = solutions[x1]
            x2_val = solutions[x2]
        
        H_val = hess.subs({x1: x1_val, x2: x2_val})
        
        print(f"\nDurağan noktada Hessian matrisi:")
        print(H_val)
        
        # Özdeğerleri hesapla
        eigenvals = list(H_val.eigenvals().keys())
        print(f"Özdeğerler: {eigenvals}")
        
        # Durağan noktanın türünü belirle
        if all(val > 0 for val in eigenvals):
            print("\nBu nokta bir minimum noktasıdır.")
        elif all(val < 0 for val in eigenvals):
            print("\nBu nokta bir maksimum noktasıdır.")
        elif any(val > 0 for val in eigenvals) and any(val < 0 for val in eigenvals):
            print("\nBu nokta bir eyer (semer) noktasıdır.")
        else:
            print("\nBu noktanın türü belirlenemedi.")

if __name__ == "__main__":
    while True:
        print("\n=== Durağan Nokta Türü Belirleme ===")
        print("1. Kendi fonksiyonunuzu analiz edin")
        print("2. Örnek çözümü göster: f(x1,x2) = (x1-1)^2 + (x2-1)^2 - x1*x2")
        print("3. Çıkış")
        
        choice = input("Seçiminiz (1/2/3): ")
        
        if choice == '1':
            determine_stationary_point_type()
        elif choice == '2':
            test_example()
        elif choice == '3':
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim, lütfen tekrar deneyin.")