#!/bin/sh

set -o errexit  # Exit the script with error if any of the commands fail
export FSLOUTPUTTYPE='NIFTI_GZ'
Path1=$1
File=$2
Path2=$3
BFC=$4
echo "processing file from ${Path1}: ${File}"

mkdir -p ${Path2}/${File}BFC/
cp ${Path1}/${File}.nii.gz ${Path2}/${File}BFC/T1.nii.gz
cd ${Path2}/${File}BFC/

if [ "$BFC" = "fast" ]; then
  ${FSLDIR}/bin/fast -B T1.nii.gz
  mv T1_restore.nii.gz ${Path2}/${File}.nii.gz
  rm -r ${Path2}/${File}BFC/
elif [ "$BFC" = "N3" ]; then
  N3BiasFieldCorrection -i T1.nii.gz -o T1_N3.nii.gz
  mv T1_N3.nii.gz ${Path2}/${File}.nii.gz
  rm -r ${Path2}/${File}BFC/
#elif [ "$BFC" = "N4" ]; then
#  ${ANTSPATH}/N4BiasFieldCorrection -i T1.nii.gz -o T1_N4.nii.gz
#  mv T1_N4.nii.gz ${Path2}/${File}.nii.gz
#  rm -r ${Path2}/${File}BFC/
else
  echo "No bias field correction method selected"
  rm -r ${Path2}/${File}/
fi

