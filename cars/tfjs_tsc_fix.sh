# https://github.com/tensorflow/tfjs/issues/1092
# Temporary fix to the issue `Cannot find module './tfjs_binding'` on tsc build.

echo "Running patch for missing tfjs-node types"
TARGET_TFJS_FILE=node_modules/@tensorflow/tfjs-node/dist/tfjs_binding.d.ts
if [ ! -f $TARGET_TFJS_FILE ]; then
  echo "export type TFEOpAttr = any; export type TFJSBinding = any;" > $TARGET_TFJS_FILE
else
  echo "> TFJS target file already exists: ${TARGET_TFJS_FILE}"
fi
