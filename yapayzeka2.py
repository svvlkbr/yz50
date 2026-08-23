"""
class Neron:
	def __init__(self,agirliklar,bias):
		self.agirliklar=agirliklar
		self.bias=bias

	def ileri_besleme(self,girdiler):
		toplam=0
		for g, a in zip(girdiler,self.agirliklar):
			toplam+=g*a
		z=toplam +self.bias

		e=2.718
		cikti=1/(1+(e**(-z)))

		return cikti

neron=Neron(agirliklar=[0.5,-0.4], bias=1.0)
input=[2.0,1.5]

sonuc=neron.ileri_besleme(input)
print(f"sonuc:{sonuc}")
"""
class Neron:
	def __init__(self,agirliklar,bias):
		self.agirliklar=agirliklar
		self.bias=bias

	def ileri_besleme(self,giris_verisi):
		toplam=0
		for g,a in zip(giris_verisi,self.agirliklar):
			toplam+=g*a
		z=toplam+self.bias

		e=2.718
		cikti=1/(1+(e**(-z)))

		return cikti

class Katman:
	def __init__(self,neron_listesi):
		self.neronlar=neron_listesi

	def ileri_besleme(self,girdiler):
		nrn_sonuclar=[]
		for neron in self.neronlar:
			sonuc=neron.ileri_besleme(girdiler)
			nrn_sonuclar.append(sonuc)
		return nrn_sonuclar

n1=Neron(agirliklar=[0.5,-0.4],bias=1.0)
n2=Neron(agirliklar=[0.1,0.2],bias=-1.0)
n3=Neron(agirliklar=[-0.3,0.8],bias=0.0)

gizli_katman=Katman([n1,n2,n3])

veri=[2.0,1.5]
katman_ciktilari=gizli_katman.ileri_besleme(veri)

print(f"Katman cıktıları:{katman_ciktilari}")



