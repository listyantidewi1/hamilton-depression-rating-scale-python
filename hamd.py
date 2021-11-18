# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:41:02 2021

@author: DELL
"""
from fbprophet import Prophet
from pylab import rcParams
import matplotlib
import statsmodels.api as sm
import pandas as pd
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

#Baca dan tampilkan CSV
df = pd.read_csv("hamd.csv")
print(df)

#Rename Kolom
df.columns = ['jenis_guru', 'jenjang', 'usia', 'jk','hamil_menyusui','pendidikan','lama_mengabdi','jml_anak_under_17','jml_anak_upper_17','jml_murid','jml_jp','jarak_rumah','skp','jml_jam_ws_reguler','jml_jam_ws_mental_health','depressed_mood','feelings_of_guilt','suicide_feelings','insomnia1','insomnia2','insomnia3','job_activity','retardation','agitation','anxiety','fis_anxiety','fis_gastro','common_somatic','genitalia','severe_illness','weight_loss','insight']

#Mapping Nilai
depressed_mood_dict = {'Tidak merasakan / absent': 0, 'Saya merasakaan sebagian perasaan tersebut sekarang setelah saya mendapat pertanyaan ini (sebelumnya tidak terpikirkan bahwa saya merasakan hal tersebut)':1, 'Saya tidak yakin tapi teman-teman di sekitar saya sering mengatakan bahwa saya terlihat murung':2, 'Ya saya merasakan sebagian/semua hal tersebut':3}
feelings_of_guilt_dict = {'Tidak merasakan / absent':0, 'Saya merasa sering mengecewakan orang lain (termasuk keluarga)':1, 'Saya terbayang-bayang kesalahan masa lalu saya dan bayangan tersebut menghantui saya dan membuat saya merasa bersalah':2, 'Rasa sakit / kesulitan-kesulitan yang saya alami sekarang merupakan hukuman atas kesalahan-kesalahan saya di masa lampau':3, 'Saya cukup yakin bahwa ada banyak orang menyalahkan saya atau menuduh saya melakukan kesalahan':4}
suicide_dict = {'Tidak merasakan / absent':0, 'Saya merasa putus asa dengan kehidupan saya':1, 'Saya berharap saya tidak pernah terlahir':2, 'Saya berharap saya mati saja':3, 'Saya sedang memikirkan untuk mencoba bunuh diri':4}
insomnia1_dict = {'Tidak ada kesulitan untuk terlelap':0, 'Kadang-kadang memerlukan waktu hingga lebih dari setengah jam hingga dapat tertidur':1, 'Sering memerlukan waktu hingga lebih dari setengah jam hingga dapat tertidur':2}
insomnia2_dict = {'Sangat jarang terbangun di tengah malam yang menyebabkan kesulitan untuk kembali tidur':0, 'Kadang-kadang terbangun di tengah malam tanpa sebab dan tujuan dan sulit untuk kembali tidur':1, 'Sering terbangun di tengah malam tanpa sebab dan tujuan dan sulit untuk kembali tidur':2}
insomnia3_dict = {'Sangat jarang terbangun dini hari':0, 'Kadang-kadang terbangun dini hari tanpa sebab dan tujuan dan sulit untuk tidur kembali':1, 'Sering terbangun dini hari tanpa sebab dan tujuan dan sulit untuk tidur kembali':2}
job_activity_dict = {'Saya tidak mengalami kesulitan yang berarti dalam pekerjaan dan aktivitas rutin saya':0, 'Saya merasa tidak kompeten dalam pekerjaan saya / sering merasakan kelelahan atau lemas yang menurut saya ada kaitannya dengan pekerjaan dan aktivitas saya':1, 'Saya merasa kurang tertarik dan kurang semangat untuk bekerja dan beraktivitas':2, 'Saya menghabiskan waktu kurang dari 4 jam untuk benar-benar bekerja dan beraktivitas rutin':3,'Saya berhenti bekerja karena sudah tidak tahan lagi':4}
retardation_dict = {'Saya tidak mengalami perlambatan':0, 'Saya sepertinya mengalami sedikit perlambatan':1, 'Saya yakin bahwa saya mengalami perlambatan':2, 'Kegiatan semacam interview sangat sulit bagi saya dan saya tidak tau harus bicara apa dan harus bagaimana':3}
agitation_dict = {'Tidak mengalami agitasi':0, 'Saya merasa resah':1, 'Orang sering mengatakan bahwa saya punya kebiasaan memainkan rambut / menggigit kuku dan sejenisnya tanpa sadar':2, 'Saya tidak bisa duduk tenang':3}
anxiety_dict = {'Tidak merasakan kecemasan':0,'Saya merasa mudah tersinggung dan mudah tersulut emosi':1,'Saya sering merasa cemas terhadap hal-hal sepele':2,'Saya merasakan ketakutan ':3}
fis_anxiety_dict = {'Tidak mengalami':0,'Mengalami gejala ringan':1,'Mengalami gejala sedang':2,'Mengalami gejala parah':3}
fis_gastro_dict = {'Tidak ada':0,'Penurunan nafsu makan':1,'Mengalami gangguan makan (membutuhkan pengobatan)':2}
common_somatic_dict = {'Tidak ada':0,'Badan terasa berat untuk digerakkan / sakit punggung / sakit pinggang / kehilangan energi':1}
genitalia_dict = {'Tidak ada':0, 'Ringan':1, 'Sedang':2, 'Berat':3}
severe_illness_dict = {'Tidak merasa':0,'Merasa ada penyakit berat pada tubuh':1,'Saya memang didiagnosa penyakit berat oleh dokter':2,'Merasa memiliki penyakit berat dan sedang mencari pengobatan':3}
weight_loss_dict = {'Tidak mengalami penurunan berat badan':0,'Berat badan saya sepertinya menurun':1,'Berat badan saya jelas-jelas menurun':2, 'Tidak pernah mengukur':3}
insight_dict = {'Saya sadar bahwa saya depresi':1, 'Saya merasa sedang sakit tapi bukan depresi. Mungkin karena pola makan dan pola tidur':2, 'Saya tidak merasa depresi dan tidak merasa sakit':3}


df['depressed_mood'] = df.depressed_mood.map(depressed_mood_dict)
df['feelings_of_guilt'] = df.feelings_of_guilt.map(feelings_of_guilt_dict)
df['suicide_feelings'] = df.suicide_feelings.map(suicide_dict)
df['insomnia1'] = df.insomnia1.map(insomnia1_dict)
df['insomnia2'] = df.insomnia2.map(insomnia2_dict)
df['insomnia3'] = df.insomnia3.map(insomnia3_dict)
df['job_activity'] = df.job_activity.map(job_activity_dict)
df['retardation'] = df.retardation.map(retardation_dict)
df['agitation'] = df.agitation.map(agitation_dict)
df['anxiety'] = df.anxiety.map(anxiety_dict)
df['fis_anxiety'] = df.fis_anxiety.map(fis_anxiety_dict)
df['fis_gastro'] = df.fis_gastro.map(fis_gastro_dict)
df['common_somatic'] = df.common_somatic.map(common_somatic_dict)
df['genitalia'] = df.genitalia.map(genitalia_dict)
df['severe_illness'] = df.severe_illness.map(severe_illness_dict)
df['weight_loss'] = df.weight_loss.map(weight_loss_dict)
df['insight'] = df.insight.map(insight_dict)


#Ganti NaN dengan 0
df['depressed_mood'] = df['depressed_mood'].fillna(0)
df['feelings_of_guilt'] = df['feelings_of_guilt'].fillna(0)
df['suicide_feelings'] = df['suicide_feelings'].fillna(0)
df['insomnia1'] = df['insomnia1'].fillna(0)
df['insomnia2'] = df['insomnia2'].fillna(0)
df['insomnia3'] = df['insomnia3'].fillna(0)
df['job_activity'] = df['job_activity'].fillna(0)
df['retardation'] = df['retardation'].fillna(0)
df['agitation'] = df['agitation'].fillna(0)
df['anxiety'] = df['anxiety'].fillna(0)
df['fis_anxiety'] = df['fis_anxiety'].fillna(0)
df['fis_gastro'] = df['fis_gastro'].fillna(0)
df['common_somatic'] = df['common_somatic'].fillna(0)
df['genitalia'] = df['genitalia'].fillna(0)
df['severe_illness'] = df['severe_illness'].fillna(0)
df['weight_loss'] = df['weight_loss'].fillna(0)
df['insight'] = df['insight'].fillna(0)


#df.columns = ['jenis_guru', 'jenjang', 'usia', 'jk','hamil_menyusui','pendidikan','lama_mengabdi','jml_anak_under_17','jml_anak_upper_17','jml_murid','jml_jp','jarak_rumah','skp','jml_jam_ws_reguler','jml_jam_ws_mental_health','depressed_mood','feelings_of_guilt','suicide_feelings','insomnia1','insomnia2','insomnia3','job_activity','retardation','agitation','anxiety','fis_anxiety','fis_gastro','common_somatic','genitalia','severe_illness','weight_loss','insight']

#Hitung score HamD pada kolom tertentu
col_list = list(df)
col_list.remove('jenis_guru')
col_list.remove('jenjang')
col_list.remove('usia')
col_list.remove('jk')
col_list.remove('jml_jp')
col_list.remove('hamil_menyusui')
col_list.remove('pendidikan')
col_list.remove('lama_mengabdi')
col_list.remove('jml_anak_under_17')
col_list.remove('jml_anak_upper_17')
col_list.remove('jml_murid')
col_list.remove('jarak_rumah')
col_list.remove('skp')
col_list.remove('jml_jam_ws_reguler')
col_list.remove('jml_jam_ws_mental_health')

df['hamd_score'] = df[col_list].sum(axis=1)

#Dekomposisi
GuruPNS = df.loc[df['jenis_guru'] == 'Guru PNS']
GuruHonorer = df.loc[df['jenis_guru']=='Guru Honorer']
GuruTY = df.loc[df['jenis_guru']=='Guru Tetap Yayasan']
GuruP3K = df.loc[df['jenis_guru']=='Guru PPPK']

GuruSD = df.loc[df['jenjang']=='SD / MI'] 
GuruSMA = df.loc[df['jenjang']=='SMA / MA']
GuruSMK = df.loc[df['jenjang']=='SMK / MAK'] 

GuruP = df.loc[df['jk']=='Perempuan']
GuruL = df.loc[df['jk']=='Laki-laki']
GuruHamil = df.loc[df['hamil_menyusui']=='Ya']


#Histogram & BoxPlot

df['hamd_score'].hist(edgecolor='yellow')
plt.title('Histogram Score Ham-D Guru Keseluruhan')
plt.show()

df['hamd_score'].plot.box(figsize=(8,6))
plt.title('Box Plot Ham-D Score Guru Keseluruhan')
plt.show()

GuruPNS['hamd_score'].hist(edgecolor='yellow')
plt.title('Histogram Score Ham-D Guru PNS')
plt.show()

GuruPNS['hamd_score'].plot.box(figsize=(8,6))
plt.title('Box Plot Ham-D Score Guru PNS')
plt.show()

GuruHonorer['hamd_score'].hist(edgecolor='yellow')
plt.title('Histogram Score Ham-D Guru Honorer')
plt.show()

GuruHonorer['hamd_score'].plot.box(figsize=(8,6))
plt.title('Box Plot Ham-D Score Guru Honorer')
plt.show()

GuruTY['hamd_score'].hist(edgecolor='yellow')
plt.title('Histogram Score Ham-D Guru Tetap Yayasan')
plt.show()

GuruTY['hamd_score'].plot.box(figsize=(8,6))
plt.title('Box Plot Ham-D Score Guru Tetap Yayasan')
plt.show()

GuruP3K['hamd_score'].hist(edgecolor='yellow')
plt.title('Histogram Score Ham-D Guru PPPK')
plt.show()

GuruP3K['hamd_score'].plot.box(figsize=(8,6))
plt.title('Box Plot Ham-D Score Guru PPPK')
plt.show()

GuruP['hamd_score'].hist(edgecolor='yellow')
plt.title('Histogram Score Ham-D Guru Perempuan Keseluruhan')
plt.show()

GuruP['hamd_score'].plot.box(figsize=(8,6))
plt.title('Box Plot Ham-D Score Guru Perempuan Keseluruhan')
plt.show()

GuruL['hamd_score'].hist(edgecolor='yellow')
plt.title('Histogram Score Ham-D Guru Laki-laki Keseluruhan')
plt.show()

GuruL['hamd_score'].plot.box(figsize=(8,6))
plt.title('Box Plot Ham-D Score Guru Laki-laki Keseluruhan')
plt.show()


#Histogram Berdasarkan kolom tertentu

df.hist(column='hamd_score',by='jk',edgecolor='yellow')
plt.suptitle('Histogram Score Ham-D Guru Berdasarkan Jenis Kelamin', x=0.5, y=1.05, ha='center', fontsize='xx-large')
plt.text(0.5, 0.04, 'Score Ham-D', ha='center')
plt.text(0.04, 0.5, 'Jumlah', va='center', rotation='vertical')
plt.show()

df.hist(column='hamd_score',by='jenjang',edgecolor='yellow')
plt.suptitle('Histogram Score Ham-D Guru Berdasarkan Jenjang Mengajar', x=0.5, y=1.05, ha='center', fontsize='xx-large')
plt.text(0.5, 0.04, 'Score Ham-D', ha='center')
plt.text(0.04, 0.5, 'Jumlah', va='center', rotation='vertical')
plt.show()

GuruP.hist(column='hamd_score',by='hamil_menyusui',edgecolor='yellow')
plt.suptitle('Histogram Score Ham-D Guru Perempuan Berdasarkan Kondisi Hamil/Menyusui', x=0.5, y=1.05, ha='center', fontsize='xx-large')
plt.text(0.5, 0.04, 'Score Ham-D', ha='center')
plt.text(0.04, 0.5, 'Jumlah', va='center', rotation='vertical')
plt.show()

df.hist(column='hamd_score',by='jenis_guru',edgecolor='yellow')
plt.suptitle('Histogram Score Ham-D Guru Berdasarkan Jenis Guru', x=0.5, y=1.05, ha='center', fontsize='xx-large')
plt.text(0.5, 0.04, 'Score Ham-D', ha='center')
plt.text(0.04, 0.5, 'Jumlah', va='center', rotation='vertical')
plt.show()

plt.hist(GuruP['hamd_score'], label='Perempuan', alpha=.8, edgecolor='red')
plt.hist(GuruL['hamd_score'], label='Laki-Laki', alpha=0.7, edgecolor='yellow')
plt.suptitle('Histogram Score Ham-D Guru Berdasarkan Jenis Kelamin Guru', x=0.5, y=1.05, ha='center', fontsize='xx-large')
plt.legend()
plt.show()

plt.hist(GuruSMK['hamd_score'], label='Guru SMK', alpha=.8, edgecolor='red')
plt.hist(GuruSD['hamd_score'], label='Guru SD', alpha=0.7, edgecolor='yellow')
plt.suptitle('Histogram Score Ham-D Guru SMK dan Guru SD', x=0.5, y=1.05, ha='center', fontsize='xx-large')
plt.legend()
plt.show()

plt.hist(GuruP['hamd_score'], label='Guru Perempuan', alpha=.8, edgecolor='red')
plt.hist(GuruHamil['hamd_score'], label='Guru Hamil / Menyusui', alpha=0.7, edgecolor='yellow')
plt.suptitle('Histogram Score Ham-D Guru Perempuan dan Guru Hamil/Menyusui', x=0.5, y=1.05, ha='center', fontsize='xx-large')
plt.legend()
plt.show()

data = [GuruP['hamd_score'], GuruL['hamd_score']]
fig = plt.figure(figsize =(10, 15))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
 
# show plot
plt.suptitle('BoxPlot Ham-D Score Guru Laki-Laki vs Guru Perempuan')
plt.legend()
plt.show()


data = [GuruP['hamd_score'], GuruHamil['hamd_score']]
fig = plt.figure(figsize =(10, 15))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
 
# show plot
plt.suptitle('BoxPlot Ham-D Score Guru Perempuan vs Guru Hamil')
plt.legend()
plt.show()


data = [GuruSMK['hamd_score'], GuruSD['hamd_score']]
fig = plt.figure(figsize =(10, 15))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
 
# show plot
plt.suptitle('BoxPlot Ham-D Score Guru SMK vs Guru SD')
plt.legend()
plt.show()

#Menghitung korelasi
#Korelasi score Ham-D dengan Jumlah jam mengajar
print("Korelasi Score Ham-D dengan Jumlah jam mengajar")
corr_hamd_jml_jp = np.corrcoef(df['hamd_score'], df['jml_jp'])
print(corr_hamd_jml_jp)
plt.scatter(df['hamd_score'],df['jml_jp'])
plt.title('Korelasi Score Ham-D Guru dengan Jumlah Jam Mengajar ')

plt.xlabel('Ham-D Score')
plt.ylabel('Jumlah Jam Mengajar')
plt.show()


print("Korelasi Score Ham-D dengan Lama Mengabdi")
corr_hamd_lama_mengabdi = np.corrcoef(df['hamd_score'], df['lama_mengabdi'])
print(corr_hamd_lama_mengabdi)
plt.scatter(df['hamd_score'],df['lama_mengabdi'])
plt.title('Korelasi Score Ham-D Guru dengan Lama Mengabdi ')

plt.xlabel('Ham-D Score')
plt.ylabel('Lama Mengabdi')
plt.show()

print("Korelasi Score Ham-D dengan Usia")
corr_hamd_usia = np.corrcoef(df['hamd_score'], df['usia'])
print(corr_hamd_usia)
plt.scatter(df['hamd_score'],df['usia'])
plt.title('Korelasi Score Ham-D Guru dengan Usia')

plt.xlabel('Ham-D Score')
plt.ylabel('Usia')
plt.show()

print("Korelasi Score Ham-D dengan Jumlah Murid")
corr_hamd_jml_murid = np.corrcoef(df['hamd_score'], df['jml_murid'])
print(corr_hamd_jml_murid)
plt.scatter(df['hamd_score'],df['jml_murid'])
plt.title('Korelasi Score Ham-D Guru dengan Jumlah Murid')

plt.xlabel('Ham-D Score')
plt.ylabel('Jumlah Murid')
plt.show()

print("Korelasi Score Ham-D dengan Jarak Rumah")
corr_hamd_jarak_rumah = np.corrcoef(df['hamd_score'], df['jarak_rumah'])
print(corr_hamd_jarak_rumah)
plt.scatter(df['hamd_score'],df['jarak_rumah'])
plt.title('Korelasi Score Ham-D Guru dengan Jarak Rumah')

plt.xlabel('Ham-D Score')
plt.ylabel('Jarak Rumah')
plt.show()


print("Korelasi Score Ham-D dengan Jumlah Jam Mengikuti Workshop Non Mental Health")
corr_hamd_jarak_rumah = np.corrcoef(df['hamd_score'], df['jml_jam_ws_reguler'])
print(corr_hamd_jarak_rumah)
plt.scatter(df['hamd_score'],df['jml_jam_ws_reguler'])
plt.title('Korelasi Score Ham-D dengan Jumlah Jam Mengikuti Workshop Non Mental Health')

plt.xlabel('Ham-D Score')
plt.ylabel('Jumlah Jam Mengikuti Workshop Non Mental Health')
plt.show()


print("Korelasi Score Ham-D dengan Jumlah Jam Mengikuti Workshop Mental Health")
corr_hamd_jarak_rumah = np.corrcoef(df['hamd_score'], df['jml_jam_ws_mental_health'])
print(corr_hamd_jarak_rumah)
plt.scatter(df['hamd_score'],df['jml_jam_ws_mental_health'])
plt.title('Korelasi Score Ham-D dengan Jumlah Jam Mengikuti Workshop Mental Health')

plt.xlabel('Ham-D Score')
plt.ylabel('Jumlah Jam Mengikuti Workshop Mental Health')
plt.show()


#Menghitung Means, Medians, Standard Deviation
rata2_hamd_all = np.mean(df['hamd_score'])
rata2_hamd_pns = np.mean(GuruPNS['hamd_score'])
rata2_hamd_gtt = np.mean(GuruHonorer['hamd_score'])
rata2_hamd_gty = np.mean(GuruTY['hamd_score'])
rata2_hamd_p3k = np.mean(GuruP3K['hamd_score'])
rata2_hamd_p = np.mean(GuruP['hamd_score'])
rata2_hamd_l = np.mean(GuruL['hamd_score'])
rata2_hamd_hm = np.mean(GuruHamil['hamd_score'])
rata2_hamd_smk = np.mean(GuruSMK['hamd_score'])
rata2_hamd_sd = np.mean(GuruSD['hamd_score'])

"""print("Rata-Rata Ham-D Score Seluruh Guru",rata2_hamd_all)
print("Rata-Rata Ham-D Score Guru PNS",rata2_hamd_pns)
print("Rata-Rata Ham-D Score Guru Honorer",rata2_hamd_gtt)
print("Rata-Rata Ham-D Score Guru Tetap Yayasan",rata2_hamd_gty)
print("Rata-Rata Ham-D Score Guru PPPK",rata2_hamd_p3k)
print("Rata-Rata Ham-D Score Guru Perempuan",rata2_hamd_p)
print("Rata-Rata Ham-D Score Guru Laki-laki",rata2_hamd_l)
print("Rata-Rata Ham-D Score Guru Hamil",rata2_hamd_hm)
print("Rata-Rata Ham-D Score Guru SMK",rata2_hamd_smk)
print("Rata-Rata Ham-D Score Guru SD",rata2_hamd_sd)"""

med_hamd_all = np.median(df['hamd_score'])
med_hamd_pns = np.median(GuruPNS['hamd_score'])
med_hamd_gtt = np.median(GuruHonorer['hamd_score'])
med_hamd_gty = np.median(GuruTY['hamd_score'])
med_hamd_p3k = np.median(GuruP3K['hamd_score'])
med_hamd_p = np.median(GuruP['hamd_score'])
med_hamd_l = np.median(GuruL['hamd_score'])
med_hamd_hm = np.median(GuruHamil['hamd_score'])
med_hamd_smk = np.median(GuruSMK['hamd_score'])
med_hamd_sd = np.median(GuruSD['hamd_score'])

"""
print("Median Ham-D Score Seluruh Guru",med_hamd_all)
print("Median Ham-D Score Guru PNS",med_hamd_pns)
print("Median Ham-D Score Guru Honorer",med_hamd_gtt)
print("Median Ham-D Score Guru Tetap Yayasan",med_hamd_gty)
print("Median Ham-D Score Guru PPPK",med_hamd_p3k)
print("Median Ham-D Score Guru Perempuan",med_hamd_p)
print("Median Ham-D Score Guru Laki-laki",med_hamd_l)
print("Median Ham-D Score Guru Hamil",med_hamd_hm)
print("Median Ham-D Score Guru SMK",med_hamd_smk)
print("Median Ham-D Score Guru SD",med_hamd_sd)"""

std_hamd_all = np.std(df['hamd_score'])
std_hamd_pns = np.std(GuruPNS['hamd_score'])
std_hamd_gtt = np.std(GuruHonorer['hamd_score'])
std_hamd_gty = np.std(GuruTY['hamd_score'])
std_hamd_p3k = np.std(GuruP3K['hamd_score'])
std_hamd_p = np.std(GuruP['hamd_score'])
std_hamd_l = np.std(GuruL['hamd_score'])
std_hamd_hm = np.std(GuruHamil['hamd_score'])
std_hamd_smk = np.std(GuruSMK['hamd_score'])
std_hamd_sd = np.std(GuruSD['hamd_score'])

"""
print("Standar Deviasi Ham-D Score Seluruh Guru",std_hamd_all)
print("Standar Deviasi Ham-D Score Guru PNS",std_hamd_pns)
print("Standar Deviasi Ham-D Score Guru Honorer",std_hamd_gtt)
print("Standar Deviasi Ham-D Score Guru Tetap Yayasan",std_hamd_gty)
print("Standar Deviasi Ham-D Score Guru PPPK",std_hamd_p3k)
print("Standar Deviasi Ham-D Score Guru Perempuan",std_hamd_p)
print("Standar Deviasi Ham-D Score Guru Laki-laki",std_hamd_l)
print("Standar Deviasi Ham-D Score Guru Hamil",std_hamd_hm)
print("Standar Deviasi Ham-D Score Guru SMK",std_hamd_smk)
print("Standar Deviasi Ham-D Score Guru SD",std_hamd_sd)"""

data = {'Keterangan':['Keseluruhan','Guru PNS','Guru Honorer','Guru Tetap Yayasan','Guru PPPK','Guru Perempuan','Guru Laki-laki','Guru Hamil/Menyusui','Guru SMK','Guru SD'],'Nilai Rata-Rata':[rata2_hamd_all,rata2_hamd_pns,rata2_hamd_gtt,rata2_hamd_gty,rata2_hamd_p3k,rata2_hamd_p,rata2_hamd_l,rata2_hamd_hm,rata2_hamd_smk,rata2_hamd_sd],'Median':[med_hamd_all,med_hamd_pns,med_hamd_gtt,med_hamd_gty,med_hamd_p3k,med_hamd_p,med_hamd_l,med_hamd_hm,med_hamd_smk,med_hamd_sd],'Standard Deviasi':[std_hamd_all,std_hamd_pns,std_hamd_gtt,std_hamd_gty,std_hamd_p3k,std_hamd_p,std_hamd_l,std_hamd_hm,std_hamd_smk,std_hamd_sd]}

df1 = pd.DataFrame(data)

print(df1)