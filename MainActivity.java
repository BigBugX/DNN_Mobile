package willpcvg.appv0;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.icu.text.SimpleDateFormat;
import android.media.Image;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toolbar;
//import android.support.design.widget.*;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.channels.InterruptedByTimeoutException;
//import javax.imageio.ImageIO;
import java.util.Arrays;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Locale;

//import ImageUtils;

public class MainActivity extends AppCompatActivity {


    static {
        System.loadLibrary("tensorflow_inference");
    }

    private String MODEL_PATH = "file:///android_asset/frozen_model.pb";
    private String INPUT_NAME = "input_image";
    private String OUTPUT_NAME = "inference/prediction";

    private String MODEL_PATH_DE = "file:///android_asset/test_de.pb";
    private String INPUT_NAME_DE = "input_image";
    private String OUTPUT_NAME_DE = "inference/refine/prediction";

    private String MODEL_PATH_DL = "file:///android_asset/exportpb";
    private String INPUT_NAME_DL = "ImageTensor";
    private String OUTPUT_NAME_DL = "SemanticPredictions";

    private TensorFlowInferenceInterface tf;
    private TensorFlowInferenceInterface tf2;

    float[] PREDICTIONS = new float[800 * 600];
    static FloatBuffer floatBuf = FloatBuffer.allocate(800 * 600);
    private float[] floatValues;
    private float[] floatValues_org;
    private float[] floatMask;
    private float[] floatMaskNega;
    long[] intFetcher = new long[800 * 600];
    float[] intFetcher_de = new float[114 * 152]; //PREDICTIONS_DE
    float[] intFetcher_dl = new float[513 * 513]; //PREDICTIONS_DL

    float[] org_mask = new float[800 * 600];
    float[] org_dmat = new float[800 * 600];
    float[] rel_dmat = new float[800 * 600];
    float[] org_img = new float[800 * 600 * 3];
    float[] blr_img = new float[800 * 600 * 3];

    //test RGB
    Bitmap bitmapR;
    Bitmap bitmapG;
    Bitmap bitmapB;

    private float[] floatValues_de;
    private float[] floatValues_dl;
    private float[] floatDepthMap;
    private float[] floatMattingMask;

    boolean vertical = true;

    ImageView imageView;
    TextView resultView;
    Snackbar progressBar;

    public static final int TAKE_PHOTO = 1;
    private ImageView picture;
    private Uri imageUri;

    String dir = Environment.getExternalStorageDirectory().getAbsolutePath()+"/Download/";
    String fileName = "demo_image";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tf = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH_DL);
        tf2 = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH_DE);

        Button takePhoto = (Button) findViewById(R.id.take_photo);
        picture = (ImageView) findViewById(R.id.picture);
        takePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //new a File Object, store the picture you take
                File outputImage = new File(getExternalCacheDir(), "output_image.jpg");
                try{
                    if (outputImage.exists()){
                        outputImage.delete();
                    }
                    outputImage.createNewFile();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                if (Build.VERSION.SDK_INT >= 24) {
                    imageUri = FileProvider.getUriForFile(MainActivity.this,
                            "willpcvg.appv0.fileprovider", outputImage);
                } else {
                    imageUri = Uri.fromFile(outputImage);
                }
                // Start up the Camera
                Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
                intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
                startActivityForResult(intent, TAKE_PHOTO);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        switch (requestCode) {
            case TAKE_PHOTO:
                if (resultCode == RESULT_OK) {
                    try {
                        //display the photo token before
                        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                        Bitmap bitmap2 = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                        Bitmap bitmapRelDmat;
                        Bitmap bitmapOrgMask;
                        Bitmap bitmap3;
                        //floatValues_test = ImageUtils.testTrans(bitmap);
                        //bitmap = bmFromFloatTest(floatValues_test, bitmap.getHeight(), bitmap.getWidth());
                        //bitmap = predict(bitmap);
                        bitmap = portraitMattingDL(bitmap); /* This is for portrait segmentation*/
                        bitmap2 = depthMapGen(bitmap2);      /* This is for depthMap generation */
                        org_mask = getFloatOrgMask(bitmap, 800 * 600);
                        org_dmat = getFloatOrgDmat(bitmap2, 800 * 600);
                        rel_dmat = getRelDmat(org_mask, org_dmat, 800 * 600);
                        floatMattingMask = triChannelFromOne(rel_dmat, 800, 600);
                        bitmapRelDmat = bmFromFloatTest(floatMattingMask, 800, 600);
                        floatMattingMask = triChannelFromOne(org_mask, 800, 600);
                        bitmapOrgMask = bmFromFloatTest(floatMattingMask, 800, 600);

                        blr_img = imgBlur(org_img, rel_dmat, 800*600, vertical);
                        bitmap3 = bmFromFloatTest(org_img, 800, 600);
                        if (vertical){
                            bitmap = bmFromFloatTest(blr_img, 800, 600);
                        } else {
                            bitmap = bmFromFloatTest(blr_img, 600, 800);
                        }

                        //Write file
                        try{
                            File file = new File(dir + fileName + "1_blur.jpg");
                            FileOutputStream out = new FileOutputStream(file);
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                            out.flush();
                            out.close();
                        } catch (Exception e){
                            e.printStackTrace();
                        }
                        try{
                            File file = new File(dir + fileName + "2_org.jpg");
                            FileOutputStream out = new FileOutputStream(file);
                            bitmap3.compress(Bitmap.CompressFormat.JPEG, 100, out);
                            out.flush();
                            out.close();
                        } catch (Exception e){
                            e.printStackTrace();
                        }
                        try{
                            File file = new File(dir + fileName + "3_mask.jpg");
                            FileOutputStream out = new FileOutputStream(file);
                            bitmapOrgMask.compress(Bitmap.CompressFormat.JPEG, 100, out);
                            out.flush();
                            out.close();
                        } catch (Exception e){
                            e.printStackTrace();
                        }
                        try{
                            File file = new File(dir + fileName + "4_rdmat.jpg");
                            FileOutputStream out = new FileOutputStream(file);
                            bitmapRelDmat.compress(Bitmap.CompressFormat.JPEG, 100, out);
                            out.flush();
                            out.close();
                        } catch (Exception e){
                            e.printStackTrace();
                        }
                        //picture.setImageBitmap(bitmap2);
                        picture.setImageBitmap(bitmap);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                break;
            default:
                break;
        }
    }

    public Bitmap portraitMattingDL(final Bitmap bitmap){
        int org_height = bitmap.getHeight();
        int org_width = bitmap.getWidth();

        if (org_height < org_width){
            vertical = false;
        }

        Bitmap resized_orgImg = ImageUtils.processBitmap(bitmap, 600, 800);
        if (vertical){
            org_img = ImageUtils.normalizeBitmap_dl(resized_orgImg, 800, 600);
        } else {
            org_img = ImageUtils.normalizeBitmap_dl(resized_orgImg, 600, 800);
        }
        Bitmap resized_image = ImageUtils.processBitmap(bitmap, 513, 513);
        floatValues_dl = ImageUtils.normalizeBitmap_dl(resized_image, 513, 513);
        tf.feed(INPUT_NAME_DL, floatValues_dl, 1, 513, 513, 3);
        tf.run(new String[]{OUTPUT_NAME_DL});
        tf.fetch(OUTPUT_NAME_DL, intFetcher_dl);
        floatValues_dl = portrait_label(intFetcher_dl, 513 * 513);
        floatMattingMask = triChannelFromOne(floatValues_dl, 513, 513);
        resized_image = bmFromFloatTest(floatMattingMask, 513, 513);
        if (bitmap.getWidth() < bitmap.getHeight()) {
            resized_image = ImageUtils.processBitmap(resized_image, 600, 800);
        } else {
            resized_image = ImageUtils.processBitmap(resized_image, 800, 600);
        }
        return resized_image;
    }

    public Bitmap depthMapGen(final Bitmap bitmap){
        Bitmap resized_image = ImageUtils.processBitmap(bitmap, 304, 228);
        floatValues_de = ImageUtils.normalizeBitmap_de(resized_image, 228, 304);
        tf2.feed(INPUT_NAME_DE, floatValues_de, 1, 228, 304, 3);
        tf2.run(new String[]{OUTPUT_NAME_DE});
        tf2.fetch(OUTPUT_NAME_DE, intFetcher_de);
        float t_max = maxFinder(intFetcher_de, 114*152);
        float t_min = minFinder(intFetcher_de, 114*152);
        intFetcher_de = fetcherRescaleDeN(intFetcher_de, 114*152, t_max, t_min);
        //intFetcher_de = fetcherRescaleDe(intFetcher_de, 114*152, t);
        floatDepthMap = triChannelFromOne(intFetcher_de, 114, 152);
        resized_image = bmFromFloatTest(floatDepthMap, 114, 152);
        if (bitmap.getWidth() < bitmap.getHeight()) {
            resized_image = ImageUtils.processBitmap(resized_image, 600, 800);
        } else {
            resized_image = ImageUtils.processBitmap(resized_image, 800, 600);
        }
        return resized_image;
    }

    public Bitmap deepMattingGen(final Bitmap bitmap){
        Bitmap resized_image = ImageUtils.processBitmap(bitmap, 513, 513);
        floatValues_dl = ImageUtils.normalizeBitmap_dl(resized_image, 513, 513);
        return resized_image;
    }

    public Bitmap portraitMattingD(final Bitmap bitmap){
        Bitmap resized_image = ImageUtils.processBitmap(bitmap, 600, 800);
        floatValues_org = ImageUtils.normalizeBitmap_org(resized_image, 800, 600, 1.0f);
        floatValues = ImageUtils.normalizeBitmap(resized_image, 800, 600, 255.0f);
        floatValues = padInput(floatValues, 800, 600); /* Padding into 6 channels */
        tf.feed(INPUT_NAME, floatValues, 1, 800, 600, 6);
        tf.run(new String[]{OUTPUT_NAME});
        tf.fetch(OUTPUT_NAME, intFetcher);
        intFetcher = rescaleValue(intFetcher, 800*600);
        PREDICTIONS = longToFloat(intFetcher, 800*600);
        floatMask = triChannelFromOne(PREDICTIONS, 800, 600);
        //Bitmap bm2 = bmFromFloatTest(floatMask, 800, 600);
        for(int i = 0; i < 600 * 800 * 3; i++){
            floatValues_org[i] = floatValues_org[i] * floatMask[i] / 255;
        }
        return bmFromFloatTest(floatValues_org, 800, 600);
    }


    public Bitmap portraitMatting(final Bitmap bitmap){
        //1
        Bitmap resized_image = ImageUtils.processBitmap(bitmap, 600, 800);
        Bitmap org_image = ImageUtils.processBitmap(bitmap, bitmap.getWidth(), bitmap.getHeight());
        floatValues = ImageUtils.normalizeBitmap(resized_image, 800, 600, 1.0f);
        floatValues_org = ImageUtils.normalizeBitmap_org(org_image, org_image.getHeight(), org_image.getWidth(), 1.0f);
        //2
        floatValues = padInput(floatValues, 800, 600);
        tf.feed(INPUT_NAME, floatValues, 1, 800, 600, 6);
        tf.run(new String[]{OUTPUT_NAME});
        tf.fetch(OUTPUT_NAME, intFetcher);
        intFetcher = rescaleValue(intFetcher, 800*600);
        PREDICTIONS = longToFloat(intFetcher, 800*600);
        floatValues = triChannelFromOne(PREDICTIONS, 800, 600);
        Bitmap resized_mask = bmFromFloatTest(floatValues, 800, 600);
        //3
        resized_mask = ImageUtils.processBitmap(resized_mask, bitmap.getWidth(), bitmap.getHeight());
        floatMask = ImageUtils.normalizeBitmap(resized_mask, bitmap.getHeight(),bitmap.getWidth(),1.0f);

        /*
        resized_image.recycle();
        org_image.recycle();
        resized_mask.recycle();
        resized_image = null;
        org_image = null;
        resized_mask = null;
        intFetcher = null;
        PREDICTIONS = null;
        floatValues = null;
        System.gc();*/
        for (int i = 0; i < floatMask.length; i++){
            floatValues_org[i] = floatValues_org[i] * (floatMask[i]/255);
        }
        return bmFromFloatTest(floatValues_org, bitmap.getHeight(), bitmap.getWidth());
        //4
    }



    public Bitmap predict(final Bitmap bitmap){
        Bitmap resized_image = ImageUtils.processBitmap(bitmap, 600, 800);
        floatValues = ImageUtils.normalizeBitmap(resized_image, 800, 600, 1.0f);
        floatValues = padInput(floatValues, 800, 600);
        tf.feed(INPUT_NAME, floatValues, 1, 800, 600, 6);
        tf.run(new String[]{OUTPUT_NAME});
        tf.fetch(OUTPUT_NAME, intFetcher);
        intFetcher = rescaleValue(intFetcher, 800*600);
        PREDICTIONS = longToFloat(intFetcher, 800*600);
        floatValues = triChannelFromOne(PREDICTIONS, 800, 600);
        return bmFromFloatTest(floatValues, 800, 600);
    }

    public float[] padInput(float[] orgInput, int height, int width){
        float[] output = new float[height * width * 6];
        for(int i = 0; i < (height * width); i++){
            int idx_o = i * 6;
            int idx_i = i * 3;
            output[idx_o] = orgInput[idx_i];
            output[idx_o + 1] = orgInput[idx_i + 1];
            output[idx_o + 2] = orgInput[idx_i + 2];
            //output[idx_o + 3] = orgInput[idx_i];
            //output[idx_o + 4] = orgInput[idx_i + 1];
            //output[idx_o + 5] = orgInput[idx_i + 2];
            output[idx_o + 3] = 0;
            output[idx_o + 4] = 0;
            output[idx_o + 5] = 0;
        }
        return output;
    }

    public long[] rescaleValue(long[] orgInput, int length){
        for (int i = 1; i < length; i++) {
            orgInput[i] = 255 * orgInput[i];
        }
        return orgInput;
    }


    public float[] longToFloat(long[] orgInput, int length){
        float[] output = new float[length];
        for(int i = 0; i < length; i++){
            output[i] = (float)orgInput[i];
        }
        return output;
    }

    public int[] floatToInt(float[] inputArray, int length){
        int[] outputArray = new int[length];
        for (int i = 0; i < length; i++){
            outputArray[i] = (int)inputArray[i];
        }
        return outputArray;
    }

    /*public static Bitmap bmFromFloat(float[] singleChannel, int h, int w){
        int[] output = new int[h * w];
        int[] pix = new int[h * w];
        int R, G, B;
        for (int i = 0; i < h * w; i++){
            output[i] = (int)(singleChannel[i]);
        }
        for (int y = 0; y < h; y++){
            for (int x = 0; x < w * 3; x = x + 3){
                int idx = y * w * 3 + x * 3;
                R = output[idx] & 0xff;
                G = output[idx] & 0xff;
                B = output[idx] & 0xff;
                pix[idx] = (R << 16) | (G << 8) | B;
            }
        }
        Bitmap bmp = Bitmap.createBitmap(pix, w, h, Bitmap.Config.ARGB_8888);
        return bmp;
    }*/

    public static Bitmap bmFromFloatTest(float[] triChannel, int h, int w) {
        int[] output = new int[h * w * 3];
        int[] pix = new int[h * w];
        int R, G, B;
        for (int i = 0; i < h * w * 3; i++){
            output[i] = (int)(triChannel[i]);
        }
        for (int y = 0; y < h; y++){
            for (int x = 0; x < w; x++){
                int idx = y * w * 3 + x * 3;
                R = output[idx] & 0xFF;
                G = output[idx + 1] & 0xFF;
                B = output[idx + 2] & 0xFF;
                pix[y * w + x] = (R << 16) | (G << 8) | B | 0xFF000000;
            }
        }
        Bitmap bmp = Bitmap.createBitmap(pix, w, h, Bitmap.Config.ARGB_8888);
        return bmp;
    }

    public static float[] triChannelFromOne(float[] singleChannel, int h, int w) {
        float[] output = new float[h * w * 3];
        for (int i = 0; i < h * w; i++){
            int idx = i * 3;
            output[idx] = singleChannel[i];
            output[idx + 1] = singleChannel[i];
            output[idx + 2] = singleChannel[i];
        }
        return output;
    }

    public static float maxFinder(float[] inputArray, int length) {
        float max = inputArray[0];
        for (int i = 1; i < length; i++) {
            if (inputArray[i] > max) {
                max = inputArray[i];
            }
        }
        return max;
    }

    public static float minFinder(float[] inputArray, int length) {
        float min = inputArray[0];
        for (int i = 1; i < length; i++) {
            if (inputArray[i] < min) {
                min = inputArray[i];
            }
        }
        return min;
    }
    public static float[] fetcherRescaleDe(float[] inputArray, int length, float t) {
        for (int i = 0; i < length; i++) {
            inputArray[i] = (inputArray[i] / 255.0f) * t;
        }
        return inputArray;
    }

    public static float[] fetcherRescaleDeN(float[] inputArray, int length, float max, float min) {
        float t = 255 / (max - min);
        for (int i = 0; i < length; i++) {
            inputArray[i] = (inputArray[i] - min) * t;
        }
        return inputArray;
    }

    public static float[] portrait_label(float[] inputArray, int length) {
        float[] outputArray = new float[length];
        for (int i = 0; i < length; i++) {
            if (inputArray[i] != 15){
                outputArray[i] = 0;
            }
            else {
                outputArray[i] = 255;
            }
        }
        return outputArray;
    }

    public static float[] floatFromBm(Bitmap source, int length){
        float[] output = new float[length * 3];
        int[] intValues = new int[source.getHeight() * source.getWidth()];
        source.getPixels(intValues, 0, source.getWidth(), 0, 0, source.getWidth(), source.getHeight());
        for (int i = 0; i < intValues.length; i++){
            final int val = intValues[i];
            output[i * 3] = (((val >> 16) & 0xFF) - 0);
            output[i * 3 + 1] = (((val >> 8) & 0xFF) - 0);
            output[i * 3 + 2] = ((val & 0xFF) - 0);
        }
        return output;
    }

    public static float[] getFloatOrgMask(Bitmap source_dl, int length){
        // get original matting mask
        float[] output = new float[length];
        float[] temp = new float[length * 3];
        temp = floatFromBm(source_dl, length);
        for (int i = 0; i < length; i++) {
            output[i] = temp[i * 3];
        }
        output = biFloatOrgMask(output, length);
        return output;
    }

    public static float[] getFloatOrgDmat(Bitmap source_de, int length){
        // get original depth estimation
        float[] output = new float[length];
        float[] temp = new float[length * 3];
        temp = floatFromBm(source_de, length);
        for (int i = 0; i < length; i++) {
            output[i] = temp[i * 3];
        }
        return output;
    }

    public static float[] biFloatOrgMask(float[] orgMask, int length){
        for (int i = 0; i < length; i++){
            if ((orgMask[i] != 255) && (orgMask[i] != 0)){
                orgMask[i] = 0;
            }
        }
        return orgMask;
    }

    public static float[] getRelDmat(float[] orgMask, float[] orgDmat, int length){
        float[] output = new float[length];
        int intVal;
        float floatVal;
        int mode;
        int[] counter = new int[256];
        for (int k = 0; k < 256; k++){
            counter[k] = 0;
        }
        for (int i = 0; i < length; i++){
            if (orgMask[i] == 255){
                floatVal = orgDmat[i];
                intVal = (int)floatVal;
                counter[intVal]++;
            }
        }
        mode = getMode(counter, 256);
        for (int m = 0; m < length; m++){
            if (orgMask[m] == 255){
                orgDmat[m] = mode;
            }
        }
        for (int j = 0; j < length; j++){
            if (orgDmat[j] > mode){
                output[j] = orgDmat[j] - mode;
            } else {
                output[j] = mode - orgDmat[j];
            }
        }
        float t_max = maxFinder(output, length);
        float t_min = minFinder(output, length);
        output = fetcherRescaleDeN(output, length, t_max, t_min);
        return output;
    }

    public static int getMode(int[] counter, int length){
        int maxOccur = 0;
        for (int i = 0; i < length; i++){
            if (counter[i] > maxOccur) {
                maxOccur = counter[i];
            }
        }
        for (int j = 0; j < length; j++){
            if (counter[j] == maxOccur){
                return j;
            }
        }
        return 0;
    }

    public static float[] imgBlur(float[] orgImg, float[] relDmat, int length, boolean v){
        float[] output = new float[length * 3];
        //for (int idx = 0; idx < length * 3; idx++){
        //    output[idx] = orgImg[idx];
        //}
        float[][] gau3 = new float[3][3];
        float[][] gau5 = new float[5][5];
        float[][] gau = {{0.0625f, 0.125f, 0.0625f},{0.125f, 0.25f, 0.125f},{0.0625f, 0.125f, 0.0625f}};

        if (v){
            float[][] orgR = new float[800][600];
            float[][] orgG = new float[800][600];
            float[][] orgB = new float[800][600];
            float[][] outputR = new float[800][600];
            float[][] outputG = new float[800][600];
            float[][] outputB = new float[800][600];
            for (int i = 0; i < 800; i++){
                for (int j = 0; j < 600; j++){
                    //orgR[i][j] = orgImg[i * 600 + j];
                    //orgG[i][j] = orgImg[length + i * 600 + j];
                    //orgB[i][j] = orgImg[length * 2 + i * 600 + j];
                    orgR[i][j] = orgImg[(i * 600 + j) * 3];
                    orgG[i][j] = orgImg[(i * 600 + j) * 3 + 1];
                    orgB[i][j] = orgImg[(i * 600 + j) * 3 + 2];
                    outputR[i][j] = 0;
                    outputG[i][j] = 0;
                    outputB[i][j] = 0;
                }
            }
            //filter blurring
            for (int m = 3; m < 797; m++){
                for (int n = 3; n < 597; n++){
                    if (relDmat[m * 600 + n] == 0){
                        outputR[m][n] = orgR[m][n];
                        outputG[m][n] = orgG[m][n];
                        outputB[m][n] = orgB[m][n];
                    } else if ((relDmat[m * 600 + n] > 0)&&(relDmat[m * 600 + n] <= 20)){
                        outputR[m][n] = (orgR[m-1][n-1] + orgR[m-1][n] + orgR[m-1][n+1] + orgR[m][n-1] + orgR[m][n] + orgR[m][n+1]+ orgR[m+1][n-1] + orgR[m+1][n] + orgR[m+1][n+1])/9;
                        outputG[m][n] = (orgG[m-1][n-1] + orgG[m-1][n] + orgG[m-1][n+1] + orgG[m][n-1] + orgG[m][n] + orgG[m][n+1]+ orgG[m+1][n-1] + orgG[m+1][n] + orgG[m+1][n+1])/9;
                        outputB[m][n] = (orgB[m-1][n-1] + orgB[m-1][n] + orgB[m-1][n+1] + orgB[m][n-1] + orgB[m][n] + orgB[m][n+1]+ orgB[m+1][n-1] + orgB[m+1][n] + orgB[m+1][n+1])/9;
                    } else if ((relDmat[m * 600 + n] > 20)&&(relDmat[m * 600 + n] <= 50)){
                        outputR[m][n] = (orgR[m-2][n-2] + orgR[m-2][n] + orgR[m-2][n+2] + orgR[m][n-2] + orgR[m][n] + orgR[m][n+2]+ orgR[m+2][n-2] + orgR[m+2][n] + orgR[m+2][n+2])/9;
                        outputG[m][n] = (orgG[m-2][n-2] + orgG[m-2][n] + orgG[m-2][n+2] + orgG[m][n-2] + orgG[m][n] + orgG[m][n+2]+ orgG[m+2][n-2] + orgG[m+2][n] + orgG[m+2][n+2])/9;
                        outputB[m][n] = (orgB[m-2][n-2] + orgB[m-2][n] + orgB[m-2][n+2] + orgB[m][n-2] + orgB[m][n] + orgB[m][n+2]+ orgB[m+2][n-2] + orgB[m+2][n] + orgB[m+2][n+2])/9;
                    } else if (relDmat[m * 600 + n] > 50){
                        outputR[m][n] = (orgR[m-3][n-3] + orgR[m-3][n] + orgR[m-3][n+3] + orgR[m][n-3] + orgR[m][n] + orgR[m][n+3]+ orgR[m+3][n-3] + orgR[m+3][n] + orgR[m+3][n+3])/9;
                        outputG[m][n] = (orgG[m-3][n-3] + orgG[m-3][n] + orgG[m-3][n+3] + orgG[m][n-3] + orgG[m][n] + orgG[m][n+3]+ orgG[m+3][n-3] + orgG[m+3][n] + orgG[m+3][n+3])/9;
                        outputB[m][n] = (orgB[m-3][n-3] + orgB[m-3][n] + orgB[m-3][n+3] + orgB[m][n-3] + orgB[m][n] + orgB[m][n+3]+ orgB[m+3][n-3] + orgB[m+3][n] + orgB[m+3][n+3])/9;
                    }
                }
            }

            for (int i = 0; i < 800; i++){
                for (int j = 0; j < 600; j++){
                    output[(i * 600 + j) * 3] = outputR[i][j];
                    output[(i * 600 + j) * 3 + 1] = outputG[i][j];
                    output[(i * 600 + j) * 3 + 2] = outputB[i][j];
                }
            }
            return output;
        } else {
            float[][] orgR = new float[600][800];
            float[][] orgG = new float[600][800];
            float[][] orgB = new float[600][800];
            float[][] outputR = new float[600][800];
            float[][] outputG = new float[600][800];
            float[][] outputB = new float[600][800];
            for (int i = 0; i < 600; i++){
                for (int j = 0; j < 800; j++) {
                    orgR[i][j] = orgImg[i * 800 + j];
                    orgG[i][j] = orgImg[length + i * 800 + j];
                    orgB[i][j] = orgImg[length * 2 + i * 800 + j];
                    outputR[i][j] = 0;
                    outputG[i][j] = 0;
                    outputB[i][j] = 0;
                }
            }
            //filter blurring
            for (int m = 3; m < 597; m++){
                for (int n = 3; n < 797; n++){
                    if (relDmat[m * 800 + n] == 0){
                        outputR[m][n] = orgR[m][n];
                        outputG[m][n] = orgG[m][n];
                        outputB[m][n] = orgB[m][n];
                    } else if ((relDmat[m * 600 + n] > 0)&&(relDmat[m * 600 + n] <= 20)){
                        outputR[m][n] = (orgR[m-1][n-1] + orgR[m-1][n] + orgR[m-1][n+1] + orgR[m][n-1] + orgR[m][n] + orgR[m][n+1]+ orgR[m+1][n-1] + orgR[m+1][n] + orgR[m+1][n+1])/9;
                        outputG[m][n] = (orgG[m-1][n-1] + orgG[m-1][n] + orgG[m-1][n+1] + orgG[m][n-1] + orgG[m][n] + orgG[m][n+1]+ orgG[m+1][n-1] + orgG[m+1][n] + orgG[m+1][n+1])/9;
                        outputB[m][n] = (orgB[m-1][n-1] + orgB[m-1][n] + orgB[m-1][n+1] + orgB[m][n-1] + orgB[m][n] + orgB[m][n+1]+ orgB[m+1][n-1] + orgB[m+1][n] + orgB[m+1][n+1])/9;
                    } else if ((relDmat[m * 600 + n] > 20)&&(relDmat[m * 600 + n] <= 50)){
                        outputR[m][n] = (orgR[m-2][n-2] + orgR[m-2][n] + orgR[m-2][n+2] + orgR[m][n-2] + orgR[m][n] + orgR[m][n+2]+ orgR[m+2][n-2] + orgR[m+2][n] + orgR[m+2][n+2])/9;
                        outputG[m][n] = (orgG[m-2][n-2] + orgG[m-2][n] + orgG[m-2][n+2] + orgG[m][n-2] + orgG[m][n] + orgG[m][n+2]+ orgG[m+2][n-2] + orgG[m+2][n] + orgG[m+2][n+2])/9;
                        outputB[m][n] = (orgB[m-2][n-2] + orgB[m-2][n] + orgB[m-2][n+2] + orgB[m][n-2] + orgB[m][n] + orgB[m][n+2]+ orgB[m+2][n-2] + orgB[m+2][n] + orgB[m+2][n+2])/9;
                    } else if (relDmat[m * 600 + n] > 50){
                        outputR[m][n] = (orgR[m-3][n-3] + orgR[m-3][n] + orgR[m-3][n+3] + orgR[m][n-3] + orgR[m][n] + orgR[m][n+3]+ orgR[m+3][n-3] + orgR[m+3][n] + orgR[m+3][n+3])/9;
                        outputG[m][n] = (orgG[m-3][n-3] + orgG[m-3][n] + orgG[m-3][n+3] + orgG[m][n-3] + orgG[m][n] + orgG[m][n+3]+ orgG[m+3][n-3] + orgG[m+3][n] + orgG[m+3][n+3])/9;
                        outputB[m][n] = (orgB[m-3][n-3] + orgB[m-3][n] + orgB[m-3][n+3] + orgB[m][n-3] + orgB[m][n] + orgB[m][n+3]+ orgB[m+3][n-3] + orgB[m+3][n] + orgB[m+3][n+3])/9;
                    }
                }
            }
            for (int i = 0; i < 600; i++){
                for (int j = 0; j < 800; j++){
                    output[i * 600 + j] = outputR[i][j];
                    output[length + i * 600 + j] = outputG[i][j];
                    output[length * 2 + i * 600 + j] = outputB[i][j];
                }
            }
            return output;
        }
    }

    public static float[][] seqToMat(float[] seq, int length, boolean v){
        float[][] verSeq = new float[800][600];
        float[][] horSeq = new float[600][800];
        for (int i = 0; i < 600; i++){
            for (int j = 0; j < 800; j++){
                verSeq[i][j] = seq[i * 600 + j];
            }
        }
        for (int m = 0; m < 800; m++){
            for (int n = 0; n < 600; n++){
                horSeq[m][n] = seq[m * 800 + n];
            }
        }
        if (v) {
            return verSeq;
        } else {
            return horSeq;
        }
    }
}


