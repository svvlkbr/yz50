#loss func
"""
import matplotlib.pyplot as plt
def mean_absolute_error(actual:list,predicted:list) -> float:
	if len(actual) != len(predicted):
		raise ValueError("The length of actual values and predicted values must be the same")

	absolute_diffs=[abs(act-pred) for act , pred in zip(actual,predicted)]
	mae=sum(absolute_diffs)/len(actual)

	return mae

y_true=[3,-0.5,2,7]
tahmin_senaryolari = [
    [0.0, 2.0, -1.0, 3.0],  
    [1.5, 0.5, 1.0, 5.0], 
    [2.3, -0.2, 1.8, 6.2],  
    [2.9, -0.4, 2.0, 6.9], 
]
loss_degerleri=[]

for y_pred in tahmin_senaryolari:
	mae_value=mean_absolute_error(y_true,y_pred)
	loss_degerleri.append(mae_value)
	print(f"Tahmiin:{y_pred} --> Loss (MAE):{mae_value:.4f}")

plt.figure(figsize=(8,5))
plt.plot(
	range(1,len(loss_degerleri)+1),
	loss_degerleri,
	marker="o",
	color="b",
	linestyle="-",
	linewidth=2,)
plt.title("loss eğrisi")
plt.xlabel("senaryolar")
plt.ylabel("loss değeri(mae)")
plt.grid(True)

plt.show()
"""

import matplotlib.pyplot as plt


def mean_absolute_error(actual: list, predicted: list) -> float:
  if len(actual) != len(predicted):
    raise ValueError(
        "The length of actual values and predicted values must be the same"
    )

  absolute_diffs = [abs(act - pred) for act, pred in zip(actual, predicted)]
  mae = sum(absolute_diffs) / len(actual)

  return mae


y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [0.0, 0.0, 0.0, 0.0]

learning_rate=0.1
h=0.001

loss_degerleri=[]

for adim in range(15):
	mae_value=mean_absolute_error(y_true,y_pred)
	loss_degerleri.append(mae_value)
	print(f"Adım {adim+1} | Tahmin: {[round(p, 2) for p in y_pred]} --> Loss (MAE):"
      f" {mae_value:.4f}")

	y_pred_yeni=[]
	for i in range(len(y_pred)):
		y_pred_kopya=list(y_pred)
		y_pred_kopya[i] +=h
		loss_ileri=mean_absolute_error(y_true,y_pred_kopya)

		turev=(loss_ileri-mae_value)/h

		guncel_deger=y_pred[i]-(learning_rate*turev)
		y_pred_yeni.append(guncel_deger)
	y_pred=y_pred_yeni

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(loss_degerleri) + 1),
    loss_degerleri,
    marker="o",
    color="b",
    linestyle="-",
    linewidth=2,
)
plt.title("Sayısal Türev ile Gradient Descent - Loss Eğrisi")
plt.xlabel("Eğitim Adımları (Epoch)")
plt.ylabel("Loss Değeri (MAE)")
plt.grid(True)

plt.show()