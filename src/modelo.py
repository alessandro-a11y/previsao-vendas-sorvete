import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Carregar os dados
df = pd.read_csv('../inputs/sorvetes.csv')

# 2. Separar features e target
X = df[['Temperatura']]
y = df['Vendas']

# 3. Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Fazer uma previsão
temp_teste = [[34]]
venda_prevista = modelo.predict(temp_teste)
print(f'Previsão de vendas para 34ºC: {venda_prevista[0]:.2f} sorvetes')

# 5. Visualizar
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, modelo.predict(X), color='red', label='Regressão Linear')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete')
plt.title('Temperatura vs Vendas de Sorvete')
plt.legend()
plt.grid(True)
plt.show()
