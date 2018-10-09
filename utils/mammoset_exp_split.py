# # -> Ordem recomendada de execução de experimentos
#
# - Experimento 6-1
# - Experimento 5-2
# - Experimento 5-1
# - Experimentos 3
# - Experimentos 1
# - Experimentos 2
#
#
# -> maiores informações em: https://cbms2018.hotell.kau.se/
#
# Em gbdi-mammoset-c0eca50da07e/mammoclear/label.csv, podem ser encontradas informações de cada amostra contida nos subconjuntos DDSM, MIAS e VIENNA, tais como:
#
# - nome da imagem
# - nome do subconjunto
# - possíveis classes a serem consideradas em diferentes tipos de experimentos
#
# O conjunto "gbdi-mammoset-c0eca50da07e" é composto por três subconjuntos de dados DDSM, MIAS e VIENNA. Sendo assim é possível a realização de diferentes experimentos, tais como:
#
#
# Experimentos 1 VIENNA (a partir da linha 2 até 448)
#
# Experimentos 2 MIAS (a partir da linha 449 até 566)
#
# Experimentos 3 DDSM (a partir da linha 567 até 3458)
#
# Experimentos 4 (MIAS, DDSM)
#
# Experimentos 5 (VIENNA, DDSM)
#
# Experimentos 6 (VIENNA, MIAS, DDSM)
#


import shutil
import os
import pandas as pd

mammoset_path = '/home/daniel/Downloads/CBMS/'


def normalize_filename(filename):
    f = os.path.splitext(filename)
    return f[0] + f[1].lower()


def exp_b_m():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp_b-m')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    for idx, row in df.iterrows():
        if row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'malignant')
        elif row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'benignant')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 1-1
#
# considerar apenas amostras do conjunto VIENNA
# podem ser consideradas as classes referentes ao BI-RADS (coluna C)
# nesse caso tem-se 3 classes (birads3, birads4, birads5)

def exp1_1():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp1-1')
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'Vienna']
    df['BI-RADS'] = df['BI-RADS'].astype(int).astype(str)
    for idx, row in df.iterrows():
        class_folder = os.path.join(targer_dir, 'birads' + row['BI-RADS'])
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        filename = normalize_filename(row['image_name'])
        shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                        os.path.join(class_folder, filename))


# Experimento 1-2
#
# considerar apenas amostras do conjunto VIENNA
# podem ser consideradas as classes referentes a calcification (coluna N) e mass (coluna O)
# nesse caso tem-se 3 classes (calcification, mass, normal)

def exp1_2():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp1-2')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'Vienna']
    for idx, row in df.iterrows():
        if row['calcification'] == 'T' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'calcification')
        elif row['calcification'] == 'F' and row['mass'] == 'T':
            class_folder = os.path.join(targer_dir, 'mass')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 1-3 (combinação dos experimentos 1-1 e 1-2)
#
# considerar apenas amostras do conjunto VIENNA
# podem ser consideradas as classes referentes a BI-RADS (coluna C) calcification (coluna N) e mass (coluna O)
# nesse caso tem-se 7 classes (birads3-calcification, birads4-calcification, birads5-calcification, birads3-mass, birads4-mass, birads5-mass, normal)

def exp1_3():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp1-3')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'Vienna']
    df['BI-RADS'] = df['BI-RADS'].astype(int).astype(str)
    for idx, row in df.iterrows():
        birads = 'birads' + row['BI-RADS']
        if row['calcification'] == 'T' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, birads + '-calcification')
        elif row['calcification'] == 'F' and row['mass'] == 'T':
            class_folder = os.path.join(targer_dir, birads + '-mass')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 2-1
#
# considerar apenas amostras do conjunto MIAS
# podem ser consideradas as classes referentes ao background_tissue (coluna E)
# nesse caso tem-se 3 classes (fatty glandular, fatty, dense glandular)

def exp2_1():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp2-1')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'MIAS']
    for idx, row in df.iterrows():
        bt = row['background_tissue'].lower()
        if bt == 'fatty glandular':
            class_folder = os.path.join(targer_dir, bt)
        elif bt == 'fatty':
            class_folder = os.path.join(targer_dir, bt)
        elif bt == 'dense glandular':
            class_folder = os.path.join(targer_dir, bt)
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 2-2
#
# - considerar apenas amostras do conjunto MIAS
# - podem ser consideradas as classes referentes a calcification, mass (colunas N e O)
# - nesse caso tem-se 3 classes (calcification, mass, normal)

def exp2_2():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp2-2')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'MIAS']
    for idx, row in df.iterrows():
        if row['calcification'] == 'T' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'calcification')
        elif row['calcification'] == 'F' and row['mass'] == 'T':
            class_folder = os.path.join(targer_dir, 'mass')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 2-3
#
# - considerar apenas amostras do conjunto MIAS
# - podem ser consideradas as classes referentes a calcification, mass, malignant, benignant (colunas N, O, P, Q)
# - nesse caso tem-se 5 classes (calcification-B, calcification-M, mass-B, mass-M, normal)

def exp2_3():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp2-3')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'MIAS']
    for idx, row in df.iterrows():
        if row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'calcification-M')
        elif row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'calcification-B')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'mass-M')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'mass-B')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimentos 2-4
#
# - combinação entre experimentos 2-1 e 2-2

def exp2_4():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp2-4')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'MIAS']
    for idx, row in df.iterrows():
        class_folder = os.path.join(targer_dir, 'normal')
        bt = row['background_tissue'].lower()
        if row['calcification'] == 'T' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, bt + '-calcification')
        elif row['calcification'] == 'F' and row['mass'] == 'T':
            class_folder = os.path.join(targer_dir, bt + '-mass')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimentos 2-5
#
# - combinação entre experimentos 2-1 e 2-3

def exp2_5():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp2-5')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'MIAS']
    for idx, row in df.iterrows():
        bt = row['background_tissue'].lower()
        if row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, bt + '-calcification-M')
        elif row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, bt + '-calcification-B')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, bt + '-mass-M')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, bt + '-mass-B')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 3-1
#
# considerar apenas amostras do conjunto DDSM
# podem ser consideradas as classes referentes ao BI-RADS (coluna C)
# nesse caso tem-se 3 classes (birads1, birads2, birads3)

def exp3_1():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp3-1')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'DDSM']
    df['BI-RADS'] = df['BI-RADS'].astype(int).astype(str)
    for idx, row in df.iterrows():
        class_folder = os.path.join(targer_dir, 'birads' + row['BI-RADS'])
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 3-2
#
# - considerar apenas amostras do conjunto DDSM
# - podem ser consideradas as classes referentes a calcification, mass (colunas N, O)
# - nesse caso tem-se 3 classes (calcification, mass, normal)

def exp3_2():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp3-2')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'DDSM']
    df['BI-RADS'] = df['BI-RADS'].astype(int).astype(str)
    for idx, row in df.iterrows():
        if row['calcification'] == 'T' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'calcification')
        elif row['calcification'] == 'F' and row['mass'] == 'T':
            class_folder = os.path.join(targer_dir, 'mass')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 3-3
#
# - considerar apenas amostras do conjunto DDSM
# - podem ser consideradas as classes referentes a calcification, mass, malignant, benignant (colunas N, O, P, Q)
# - nesse caso tem-se 5 classes (calcification-B, calcification-M, mass-B, mass-M, normal)

def exp3_3():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp3-3')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'DDSM']
    for idx, row in df.iterrows():
        if row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'calcification-M')
        elif row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'calcification-B')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'mass-M')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'mass-B')
        elif row['calcification'] == 'F' and row['mass'] == 'F' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimentos 3-4
#
# - combinação dos conjuntos dos experimentos 3-1 e 3-2

def exp3_4():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp3-4')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'DDSM']
    df['BI-RADS'] = df['BI-RADS'].astype(int).astype(str)
    for idx, row in df.iterrows():
        birads = 'birads' + row['BI-RADS']
        if row['calcification'] == 'T' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, birads + '-calcification')
        elif row['calcification'] == 'F' and row['mass'] == 'T':
            class_folder = os.path.join(targer_dir, birads + '-mass')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimentos 3-5
#
# - combinação dos conjuntos dos experimentos 3-1 e 3-3

def exp3_5():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp3-5')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[df['dataset'] == 'DDSM']
    df['BI-RADS'] = df['BI-RADS'].astype(int).astype(str)
    for idx, row in df.iterrows():
        birads = row['BI-RADS']
        if row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'birads' + birads + '-calcification-M')
        elif row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'birads' + birads + '-calcification-B')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'birads' + birads + '-mass-M')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'birads' + birads + '-mass-B')
        elif row['calcification'] == 'F' and row['mass'] == 'F' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 4-1
#
# - considerar apenas amostras dos conjuntos MIAS e DDSM
# - podem ser consideradas as classes referentes a calcification, mass, malignant, benignant (colunas N, O, P, Q)
# - nesse caso tem-se 5 classes (calcification-B, calcification-M, mass-B, mass-M, normal)

def exp4_1():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp4-1')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[(df['dataset'] == 'MIAS') | (df['dataset'] == 'DDSM')]
    for idx, row in df.iterrows():
        if row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'calcification-M')
        elif row['calcification'] == 'T' and row['mass'] == 'F' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'calcification-B')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'T' and row['benignant'] == 'F':
            class_folder = os.path.join(targer_dir, 'mass-M')
        elif row['calcification'] == 'F' and row['mass'] == 'T' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'mass-B')
        elif row['calcification'] == 'F' and row['mass'] == 'F' and row['malignant'] == 'F' and row['benignant'] == 'T':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 5-1
#
# - considerar apenas amostras do conjunto VIENNA e DDSM
# - podem ser consideradas as classes referentes ao BI-RADS (coluna C)
# - nesse caso tem-se 5 classes (birads1, birads2, birads3, birads4, birads5)

def exp5_1():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp5-1')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[(df['dataset'] == 'Vienna') | (df['dataset'] == 'DDSM')]
    df['BI-RADS'] = df['BI-RADS'].astype(int).astype(str)
    for idx, row in df.iterrows():
        class_folder = os.path.join(targer_dir, 'birads' + row['BI-RADS'])
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 5-2
#
# - considerar apenas amostras dos conjuntos VIENNA e DDSM
# - podem ser consideradas as classes referentes a BI-RADS, calcification, mass (colunas C, N, O)
# - nesse caso tem-se 11 classes (birads1-calcification, birads2-calcification, birads3-calcification, birads4-calcification, birads5-calcification, birads1-mass, birads2-mass, birads3-mass, birads4-mass, birads5-mass, normal)

def exp5_2():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp5-2')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    df = df.loc[(df['dataset'] == 'Vienna') | (df['dataset'] == 'DDSM')]
    df['BI-RADS'] = df['BI-RADS'].astype(int).astype(str)
    for idx, row in df.iterrows():
        birads = 'birads' + row['BI-RADS']
        if row['calcification'] == 'T' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, birads + '-calcification')
        elif row['calcification'] == 'F' and row['mass'] == 'T':
            class_folder = os.path.join(targer_dir, birads + '-mass')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


# Experimento 6-1
#
# - considerar todas as amostras dos conjuntos VIENNA, MIAS, DDSM
# - podem ser consideradas as classes referentes a calcification (coluna N) e mass (coluna O)
# - nesse caso tem-se 3 classes (calcification, mass, normal)

def exp6_1():
    origin_dir = os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/images_raw')
    targer_dir = os.path.join(mammoset_path, 'exp6-1')
    class_folder = None
    df = pd.read_csv(os.path.join(mammoset_path, 'gbdi-mammoset-c0eca50da07e/mammoclear/label.csv'))
    for idx, row in df.iterrows():
        if row['calcification'] == 'T' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'calcification')
        elif row['calcification'] == 'F' and row['mass'] == 'T':
            class_folder = os.path.join(targer_dir, 'mass')
        elif row['calcification'] == 'F' and row['mass'] == 'F':
            class_folder = os.path.join(targer_dir, 'normal')
        if class_folder is not None:
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            filename = normalize_filename(row['image_name'])
            shutil.copyfile(os.path.join(origin_dir, row['dataset'].upper(), filename),
                            os.path.join(class_folder, filename))
        class_folder = None


if __name__ == '__main__':
    exp1_1()
    exp1_2()
    exp1_3()
    exp2_1()
    exp2_2()
    exp2_3()
    exp2_4()
    exp2_5()
    exp3_1()
    exp3_2()
    exp3_3()
    exp3_4()
    exp3_5()
    exp4_1()
    exp5_1()
    exp5_2()
    exp6_1()
