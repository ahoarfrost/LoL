import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wd",type=float)
parser.add_argument("--moms",nargs=2,type=float)
args = parser.parse_args()

moms=tuple(args.moms)
wd=args.wd
print('moms:',moms, type(moms),moms[0],moms[1])
print('wd:',wd,type(wd))