import argparse
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
color_list=["#4CB391","#FF6347","#FFA500","#9400D3","#FFC0CB","#00FF00","#00FFFF","#FFD700","#FF69B4","#8FBC8F"]

def parse_args():
    parser = argparse.ArgumentParser(description="Plot a hexplot of the distance matrix")
    parser.add_argument("--reactant_csv", nargs="+", type=str, help="The csv file(s) containing the reactant distance matrix")
    parser.add_argument("--product_csv", nargs="+", type=str, help="The csv file(s) containing the product distance matrix")
    parser.add_argument("--output", type=str, help="The output file name")
    parser.add_argument("--reactant_label", type=str, help="The label for the reactant",default="reactant")
    parser.add_argument("--product_label", type=str, help="The label for the product",default="product")
    parser.add_argument("--max_dist", type=float, help="maximum distance to take into account",default=np.inf)
    parser.add_argument("--min_dist", type=float, help="minimum distance to take into account",default=0)
    args=parser.parse_args()
    assert len(args.reactant_csv) == len(args.product_csv), "The number of reactant csv files must be equal to the number of product csv files"
    assert args.output is not None, "The output file name is required"
    return args

def scatterplot(scatterplot,df,color_idx,label=None):
    if scatterplot is not None:
        # return a figure object with the scatterplot of the distance matrix
        return sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], color=color_list[color_idx], label=label)
    else:
        # add a scatterplot of the distance matrix to the figure
        return sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], color=color_list[color_idx],ax=scatterplot, label=label)

def filter_df(df,max_dist):
    return df[df.iloc[:,:] < max_dist]

def main():
    args = parse_args()
    
    ncsv=len(args.reactant_csv)

    for i in range(ncsv):
        if i == 0:
            reactant_df=pd.read_csv(args.reactant_csv[i])
            product_df=pd.read_csv(args.product_csv[i])
        else:
            reactant_df=pd.concat([reactant_df,pd.read_csv(args.reactant_csv[i])], ignore_index=True)
            product_df=pd.concat([product_df,pd.read_csv(args.product_csv[i])], ignore_index=True)

        reactant_df=filter_df(reactant_df,args.max_dist)
        product_df=filter_df(product_df,args.max_dist)

        fig, ax = plt.subplots()
        scatterplot(ax, reactant_df, 1, label=args.reactant_label)
        scatterplot(ax, product_df, 0, label=args.product_label)
        plt.xlim(args.min_dist,args.max_dist+0.01)
        plt.ylim(args.min_dist,args.max_dist+0.01)
        plt.title(f"iteration {i}")
        plt.legend()
        plt.savefig(f"{args.output}_{i}.png")
        plt.show()

if __name__ == "__main__":
    main()
