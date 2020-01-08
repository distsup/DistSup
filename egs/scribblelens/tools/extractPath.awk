
# To generate: mypath.path
# python evaluate.py egs/scribblelens/yamls/tasman.yaml runs/test  --initialize-from ../DistSup.alphabetPlusAlign/runs/test8.b1/checkpoints/best_16371_dev-cer_23.797960174842157.pkl   > out & 

# Usage: 
# -v id=<number> of path to extract. First path is id==0
# -v name=<jpg image file name>
#
# Extract 1 path
# awk -v id=4 -f extractPath.awk mypath.path  > path4.txt
# 
# Visualize 1 path
# python ./visualizeAlignment.py --path path4.txt
# 
BEGIN { 
	cnt = -1
	if (name != "") id = -1
}
/\.jpg/ {
	# Begin of path
	cnt += 1	
}
{
	if (match($1,"scribblelens") != 0)
	{
		if ((name != "") && (match($1,name) != 0))
		{
			id = cnt
		}
	}
		
	if (cnt == id)
		print $0
}