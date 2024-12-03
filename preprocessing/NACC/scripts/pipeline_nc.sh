#!/bin/sh
#this bash script use fsl to process brain MRI into MNI template
# Reference: https://github.com/vkola-lab/ncomms2022

#set -o errexit  # Exit the script with error if any of the commands fail
export FSLOUTPUTTYPE='NIFTI_GZ'
Path1=$1
File=$2
Path2=$3
echo "processing file from ${Path1}: ${File}"

mkdir -p ${Path2}/${File}/
cp ${Path1}/${File}.nii ${Path2}/${File}/
cd ${Path2}/${File}/

#step1: swap axes so that the brain is in the same direction as MNI template.
# step1：交换轴，使大脑与MNI模板处于同一方向
${FSLDIR}/bin/fslreorient2std $File.nii T1.nii.gz || ! rm -r $Path2/$File/ || exit
#${FSLDIR}/bin/fslreorient2std ${File}.nii T1.nii.gz
#if [ ! -f "T1.nii.gz" ] ; then
#  cp ${File}.nii T1.nii.gz
#fi

#step2: estimate robust field of view
# step2：估计鲁棒视场
line=`${FSLDIR}/bin/robustfov -i T1.nii.gz | grep -v Final | head -n 1`

x1=`echo ${line} | awk '{print $1}'`
x2=`echo ${line} | awk '{print $2}'`
y1=`echo ${line} | awk '{print $3}'`
y2=`echo ${line} | awk '{print $4}'`
z1=`echo ${line} | awk '{print $5}'`
z2=`echo ${line} | awk '{print $6}'`

x1=`printf "%.0f", $x1`
x2=`printf "%.0f", $x2`
y1=`printf "%.0f", $y1`
y2=`printf "%.0f", $y2`
z1=`printf "%.0f", $z1`
z2=`printf "%.0f", $z2`

#step3: cut the brain to get area of interest (roi), sometimes it cuts part of the brain
# step3：切割大脑以获得感兴趣区域（roi），有时会切割部分大脑
${FSLDIR}/bin/fslmaths T1.nii.gz -roi $x1 $x2 $y1 $y2 $z1 $z2 0 1 T1_roi.nii.gz
#cp T1.nii T1_roi.nii
# -roi <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> <tmin> <tsize> :
# zero outside roi (using voxel coordinates).
# Inputting -1 for a size will set it to the full image extent for that dimension.

#step4: remove skull -f 0.45 -g 0.1
# step4：颅骨切除，用默认参数 -f 0.5 -g 0
${FSLDIR}/bin/bet T1_roi.nii.gz T1_brain.nii.gz $4
# -f <f>  fractional intensity threshold (0->1); default=0.5;
# smaller values give larger brain outline estimates 越小整体保留越多
# -g <g>  vertical gradient in fractional intensity threshold (-1->1); default=0;
# positive values give larger brain outline at bottom, smaller at top 越小顶部保留越多，底部保留越少
#-R  robust brain centre estimation (iterates BET several times)
#-S  eye & optic nerve cleanup (can be useful in SIENA - disables -o option)
#-B  bias field & neck cleanup (can be useful in SIENA)


#step5: registration from cut to MNI
# step5：从切割（T1_brain.nii.gz）到MNI模板的配准
${FSLDIR}/bin/flirt -in T1_brain.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain \
-omat orig_to_MNI.mat

#step6: apply matrix onto original image
# step6：将矩阵应用于原始图像
${FSLDIR}/bin/flirt -in T1.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain \
-applyxfm -init orig_to_MNI.mat -out T1_MNI.nii.gz

#step7: skull remove -f 0.3 -g -0.0
# step7：颅骨切除，用自定义参数
${FSLDIR}/bin/bet T1_MNI.nii.gz T1_MNI_brain.nii.gz $5

# step8: register the skull removed scan to MNI_brain_only template again to fine tune the alignment
# step8：再次将颅骨切除后的影像 配准到 只有大脑部分的MNI模板中，以微调对齐
${FSLDIR}/bin/flirt -in T1_MNI_brain.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain -out T1_MNI_brain.nii.gz

#step9: rename and move final file
# step9：重命名并移动最终文件
mv T1_MNI_brain.nii.gz ${Path2}/${File}.nii.gz

# clear tmp folder
rm -r ${Path2}/${File}/
