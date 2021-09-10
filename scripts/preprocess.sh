#for entry in ntust/RenderPeople*/*
#do
#    python apps/obj.py -i $entry
#done

#for entry in ntust/G*/*
#do
#    python -m apps.prt_util -i $entry
#done

for entry in ntust/G*/*
do
    python -m apps.render_data -i $entry -o train -e
done
