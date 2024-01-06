import matplotlib.pyplot as plt

# Dados para o gráfico de barras
categorias = ['Categoria 1', 'Categoria 2', 'Categoria 3', 'Categoria 4']
valores = [10, 24, 15, 30]

# Criar o gráfico de barras
plt.bar(categorias, valores, color='blue')

# Adicionar rótulos e título
plt.xlabel('Categorias')
plt.ylabel('Valores')
plt.title('Gráfico de Barras Simples')

# Exibir o gráfico
plt.show()
