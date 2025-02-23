import random
from itertools import product

# Listas de palavras para o GRID em português
#[comando] [cor] [preposição] [letra] [número] [advérbio]

# 1. Comandos
comandos = [
    "apanhe",   # /aˈpa.ɲi/
    "molhe",    # /ˈmɔ.ʎi/
    "encha",    # /ˈẽ.ʃa/
    "tinja"     # /ˈtĩ.ʒa/
]

# 2. Cores
cores = [
    "vermelho",  # /veʁˈme.ʎu/
    "ferrugem",  # /feʁˈu.ʒẽj/
    "cinza",     # /ˈsĩ.za/
    "rosa"       # /ˈʁɔ.za/
]

# 3. Preposições
preposicoes = [
    "entre",    # /ˈẽ.tɾi/
    "sobre",    # /ˈso.bɾi/
    "com",      # /kõ/
    "para"      # /ˈpa.ɾa/
]

# 4. Letras
letras = [
    "A",        # á
    "B",        # bê
    "C",        # cê
    "D",        # dê
    "E",        # ê
    "F",        # éfe
    "G",        # gê
    "H",        # agá
    "I",        # i
    "J",        # jóta
    "K",        # cá
    "L",        # éle
    "M",        # ême
    "N",        # êne
    "O",        # ó
    "P",        # pê
    "Q",        # quê
    "R",        # érre
    "S",        # ésse
    "T",        # tê
    "U",        # u
    "V",        # vê
    "W",        # "dáblio" (/ˈda.bli.u/)
    "X",        # xis
    "Y",        # ípsilon
    "Z"         # zê
]

# 5. Dígitos
digitos = [
    "zero",
    "um", 
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove"
]

# 6. Advérbios
adverbios = [
    "agora",    # /aˈgɔ.ɾa/
    "já",       # /ʒa/
    "depois",   # /deˈpɔj(s)/
    "breve"     # /ˈbɾɛ.ve/
]

def gerar_frases(num_frases=500):
    """
    Gera todas as combinações possíveis, randomiza e seleciona um subconjunto
    de frases baseado na estrutura [comando] [cor] [preposição] [letra] [número] [advérbio]
    """
    # Gera todas as combinações possíveis
    todas_combinacoes = list(product(
        comandos,
        cores,
        preposicoes,
        letras,
        digitos,
        adverbios
    ))
    
    # Randomiza as combinações
    random.shuffle(todas_combinacoes)
    
    # Seleciona o número desejado de frases
    frases_selecionadas = todas_combinacoes[:num_frases]
    
    # Salva as frases em um arquivo
    with open('frases_grid.txt', 'w', encoding='utf-8') as f:
        for i, (cmd, cor, prep, letra, num, adv) in enumerate(frases_selecionadas, 1):
            frase = f"{i}. {cmd} {cor} {prep} {letra} {num} {adv}\n"
            f.write(frase)
    
    print(f"Arquivo 'frases_grid.txt' gerado com {num_frases} frases.")

if __name__ == "__main__":
    # Remove a semente fixa para ter resultados diferentes a cada execução
    gerar_frases(500)
