import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use('seaborn-white')
sns.set_style("whitegrid")

import pandas as pd

# gamma = [0], [1], [10]
# seed =
# y = np.vstack([np.random.randn(3)+50,np.random.randn(3)+5,np.random.randn(3)+10,np.random.randn(3)+15,np.random.randn(3)+20,np.random.randn(3)+30])
# shot = [1000]* 6+[2500]*6+[5000] * 6
# df_dict = {'shot':np.array(shot).reshape(-1,),'y': np.array(y).reshape(-1,), 'gamma': np.array(gamma).reshape(-1,), 'seed':np.array(seed).reshape(-1,)}
# df = pd.DataFrame(df_dict)
# fmri = sns.load_dataset("fmri")
print("a")
if 1 == 1:
    df = pd.read_csv('test_data_classifier.csv')
    # df=df.rename(columns={"gamma": "$\gamma$"})
    print(df)
    g = sns.lineplot(x='shot', y='Accuracy',
                     hue='Classifier', dashes=False, style='Classifier', markers=True, data=df, legend='full',
                     palette='deep')

    g.yaxis.set_major_formatter(mtick.PercentFormatter())
    g.set(ylabel='Top-1 Accuracy (%)')
    handles, labels = g.get_legend_handles_labels()
    g.set_yticklabels([int(y) for y in g.get_yticks()], size=14)
    g.set_xticklabels([int(y) for y in g.get_xticks()], size=14)
    g.legend(handles=handles[1:], labels=labels[1:])

    g.set_xlabel('Samples Per Class', fontsize=14)
    g.set_ylabel('Top-1 Accuracy (%)', fontsize=14)
    # g._legend.set_title("$\gamma$")
    # plt.savefig("classifiers_bench.svg",format="svg")
    plt.savefig("classifiers_bench.png")
# elif 1 == 1:
df = pd.read_csv('test_data3.csv')
# ax = sns.barplot(x='Samples Per Class', y='Accuracy',
#                   hue='Method',dashes=False,style='Method',markers=True, data=df,legend='full',palette='deep')
fig, ax1 = plt.subplots(figsize=(10, 10))

g = sns.barplot(x='shot', y='FID', hue='method', data=df, ax=ax1, palette=["slategrey", "maroon"], edgecolor=".6")
# g.set(ylabel='FID $\downarrow$',fontsize=14)
# g.set(xlabel='Samples Per Class',fontsize=14)
g.set_yticklabels([int(y) for y in g.get_yticks()], size=20)
g.set_xticklabels([10, 100, 500], size=20)

g.set_xlabel('Samples Per Class', fontsize=22)
g.set_ylabel('FID $\downarrow$', fontsize=22)
g.legend(loc='best', ncol=2, fontsize=22)

sns.despine(fig)
# g.set_title("CIFAR-100 Top-1 Accuracy")

print("done")
# plt.savefig("/fid.svg", format="svg")
plt.savefig("/fid.png")
# g.yaxis.set_major_formatter(mtick.PercentFormatter())
# g.set(ylabel='Top-1 Accuracy (%)')
# handles, labels = g.get_legend_handles_labels()
# g.legend(handles=handles[1:], labels=labels[1:])
# g._legend.set_title("$\gamma$")
# if 1 == 0:
df = pd.read_csv('test_data_cifar10.csv')
print(df)

g = sns.lineplot(x='Samples Per Class', y='Accuracy',
                 hue='method', dashes=False, style='method', markers=True, data=df, legend='full', palette='deep')
g.yaxis.set_major_formatter(mtick.PercentFormatter())

# g.set(ylabel='Top-1 Accuracy (%)')
handles, labels = g.get_legend_handles_labels()
g.set_yticklabels([int(y) for y in g.get_yticks()], size=14)
g.set_xticklabels([int(y) for y in g.get_xticks()], size=14)
g.legend(handles=handles[1:], labels=labels[1:])

g.set_xlabel('Samples Per Class', fontsize=14)
g.set_ylabel('Top-1 Accuracy (%)', fontsize=14)
# plt.savefig("cifar10_plot2.svg",format="svg")
plt.savefig("cifar10_plot.png")
