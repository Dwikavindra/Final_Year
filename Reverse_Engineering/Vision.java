package org.informatika.sirekap.support.vision;

import android.content.Context;
import android.graphics.Bitmap;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import kotlin.Metadata;
import kotlin.Pair;
import kotlin.collections.ArraysKt;
import kotlin.collections.CollectionsKt;
import kotlin.collections.IntIterator;
import kotlin.jvm.internal.DefaultConstructorMarker;
import kotlin.jvm.internal.Intrinsics;
import kotlin.ranges.RangesKt;
import kotlin.text.StringsKt;
import org.informatika.sirekap.BuildConfig;
import org.informatika.sirekap.model.Election;
import org.informatika.sirekap.model.FormConfig;
import org.informatika.sirekap.support.ElectionUtil;
import org.informatika.sirekap.support.templatematching.AprilTagConfig;
import org.informatika.sirekap.support.templatematching.ScanUtils;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

@Metadata(d1 = {"\u0000\u0001\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0002\n\u0002\u0010\u000e\n\u0000\n\u0002\u0010\u0006\n\u0000\n\u0002\u0010 \n\u0002\u0018\u0002\n\u0002\b\u0004\n\u0002\u0018\u0002\n\u0002\b\u0002\n\u0002\u0010\b\n\u0000\n\u0002\u0010\u0014\n\u0002\b\u0002\n\u0002\u0018\u0002\n\u0002\b\u0006\n\u0002\u0010\u0002\n\u0002\b\f\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0007\n\u0002\u0018\u0002\n\u0002\u0010!\n\u0002\b\u0006\n\u0002\u0018\u0002\n\u0002\b\b\n\u0002\u0018\u0002\n\u0002\b\u0003\n\u0002\u0010\u000b\n\u0002\b\u0013\n\u0002\u0018\u0002\n\u0002\b\u000b\u0018\u0000 i2\u00020\u0001:\u0002hiB\r\u0012\u0006\u0010\u0002\u001a\u00020\u0003¢\u0006\u0002\u0010\u0004J\u0010\u0010\r\u001a\u00020\b2\u0006\u0010\u000e\u001a\u00020\bH\u0002J\u0010\u0010\u000f\u001a\u00020\u00102\u0006\u0010\u0011\u001a\u00020\u0010H\u0002J\u0010\u0010\u0012\u001a\u00020\u00132\u0006\u0010\u0014\u001a\u00020\u0015H\u0002J\u0010\u0010\u0016\u001a\u00020\b2\u0006\u0010\u0011\u001a\u00020\u0010H\u0002J,\u0010\u0017\u001a\b\u0012\u0004\u0012\u00020\u00180\n2\f\u0010\u0019\u001a\b\u0012\u0004\u0012\u00020\u00180\n2\u0006\u0010\u001a\u001a\u00020\b2\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J\u0016\u0010\u001c\u001a\u00020\b2\f\u0010\u001d\u001a\b\u0012\u0004\u0012\u00020\b0\nH\u0002J\b\u0010\u001e\u001a\u00020\u001fH\u0016J>\u0010 \u001a\b\u0012\u0004\u0012\u00020\u00180\n2\u0006\u0010!\u001a\u00020\u00102\u0006\u0010\"\u001a\u00020\u00132\u0006\u0010#\u001a\u00020\u00132\u0006\u0010$\u001a\u00020\u00132\u0006\u0010%\u001a\u00020\u00132\u0006\u0010&\u001a\u00020\u0018H\u0002J(\u0010'\u001a\u00020\u001f2\u0006\u0010(\u001a\u00020\u00132\u0006\u0010)\u001a\u00020\u00132\u0006\u0010*\u001a\u00020\u00132\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J6\u0010+\u001a\b\u0012\u0004\u0012\u00020,0\n2\u0006\u0010-\u001a\u00020.2\u0006\u0010/\u001a\u00020\u00102\u0006\u00100\u001a\u00020\b2\u0006\u00101\u001a\u00020\b2\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J&\u00102\u001a\b\u0012\u0004\u0012\u00020,0\n2\u0006\u00103\u001a\u00020\u00102\u0006\u0010-\u001a\u00020.2\u0006\u0010\u001b\u001a\u00020\u0010H\u0002JF\u00104\u001a\u00020\u001f2\u0006\u00105\u001a\u0002062\f\u0010\u0019\u001a\b\u0012\u0004\u0012\u00020,072\f\u00108\u001a\b\u0012\u0004\u0012\u00020\u00180\n2\u0006\u00109\u001a\u00020\u00062\u0010\b\u0002\u0010:\u001a\n\u0012\u0004\u0012\u00020\u0018\u0018\u00010\nH\u0002J\u001c\u0010;\u001a\b\u0012\u0004\u0012\u00020\u00180\n2\f\u0010<\u001a\b\u0012\u0004\u0012\u00020\u00180\nH\u0002J\u0010\u0010=\u001a\u00020>2\u0006\u0010?\u001a\u00020\u0006H\u0002J\b\u0010@\u001a\u00020\u001fH\u0002J&\u0010A\u001a\b\u0012\u0004\u0012\u00020,0\n2\u0006\u00103\u001a\u00020\u00102\u0006\u0010-\u001a\u00020.2\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J$\u0010B\u001a\b\u0012\u0004\u0012\u00020\u0018072\f\u0010C\u001a\b\u0012\u0004\u0012\u00020\u00180\n2\u0006\u0010D\u001a\u00020\bH\u0002JF\u0010E\u001a\n\u0012\u0004\u0012\u00020\u0013\u0018\u00010\n2\u0006\u0010F\u001a\u00020G2\u0006\u0010H\u001a\u00020\u00062\u0006\u0010I\u001a\u00020\u00062\u0006\u0010J\u001a\u00020K2\u0006\u0010L\u001a\u00020K2\u0006\u0010M\u001a\u00020\u00132\u0006\u0010N\u001a\u00020\u0013J6\u0010O\u001a\u00020\u001f2\f\u0010P\u001a\b\u0012\u0004\u0012\u00020,0\n2\u0006\u00103\u001a\u00020\u00102\u0006\u0010Q\u001a\u00020\u00102\u0006\u0010R\u001a\u00020\u00132\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J\u001e\u0010S\u001a\u00020\u00102\u0006\u00103\u001a\u00020\u00102\f\u0010P\u001a\b\u0012\u0004\u0012\u00020,0\nH\u0002J~\u0010T\u001a\u00020\u001f2\u0006\u00105\u001a\u0002062\f\u0010\u0019\u001a\b\u0012\u0004\u0012\u00020,072\u0006\u0010/\u001a\u00020\u00102\u0006\u0010U\u001a\u00020\u00132\u0006\u0010V\u001a\u00020\u00132\u0006\u0010\"\u001a\u00020\u00132\u0006\u0010#\u001a\u00020\u00132\u0006\u0010$\u001a\u00020\u00132\u0006\u0010%\u001a\u00020\u00132\u0006\u00100\u001a\u00020\b2\u0006\u00101\u001a\u00020\b2\u0006\u00109\u001a\u00020\u00062\u0006\u0010W\u001a\u00020\b2\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J(\u0010X\u001a\u00020\u001f2\u0006\u0010Y\u001a\u00020,2\u0006\u00103\u001a\u00020\u00102\u0006\u0010Q\u001a\u00020\u00102\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J0\u0010Z\u001a\u00020\u001f2\u0006\u0010Y\u001a\u00020,2\u0006\u0010[\u001a\u00020\u00182\u0006\u0010Q\u001a\u00020\u00102\u0006\u0010\u001b\u001a\u00020\u00102\u0006\u0010\\\u001a\u00020,H\u0002JF\u0010]\u001a\u00020\u001f2\u0006\u0010^\u001a\u00020_2\f\u0010\u0019\u001a\b\u0012\u0004\u0012\u00020,072\u0006\u0010/\u001a\u00020\u00102\u0006\u00100\u001a\u00020\b2\u0006\u00101\u001a\u00020\b2\u0006\u00109\u001a\u00020\u00062\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J(\u0010`\u001a\u00020\u001f2\u0006\u0010Y\u001a\u00020,2\u0006\u0010a\u001a\u00020\u00182\u0006\u00103\u001a\u00020\u00102\u0006\u0010\u001b\u001a\u00020\u0010H\u0002J\u0010\u0010b\u001a\u00020\u00132\u0006\u0010\u0011\u001a\u00020\u0010H\u0002J(\u0010c\u001a\u00020\u00132\u0006\u0010d\u001a\u00020\u00102\u0006\u0010e\u001a\u00020\u00102\u0006\u0010\u001b\u001a\u00020\u00102\u0006\u0010a\u001a\u00020\u0018H\u0002J\u0016\u0010f\u001a\u00020\u00132\f\u0010g\u001a\b\u0012\u0004\u0012\u00020\b0\nH\u0002R\u000e\u0010\u0005\u001a\u00020\u0006X\u000e¢\u0006\u0002\n\u0000R\u000e\u0010\u0002\u001a\u00020\u0003X\u0004¢\u0006\u0002\n\u0000R\u000e\u0010\u0007\u001a\u00020\bX\u000e¢\u0006\u0002\n\u0000R\u0014\u0010\t\u001a\b\u0012\u0004\u0012\u00020\u000b0\nX.¢\u0006\u0002\n\u0000R\u000e\u0010\f\u001a\u00020\u000bX.¢\u0006\u0002\n\u0000¨\u0006j"}, d2 = {"Lorg/informatika/sirekap/support/vision/Vision;", "Ljava/lang/AutoCloseable;", "context", "Landroid/content/Context;", "(Landroid/content/Context;)V", "chosenMethodTabulation", "", "fontScale", "", "interpreters", "", "Lorg/tensorflow/lite/Interpreter;", "interpretersBlank", "adjustClipLimit", "contrast", "applyAdaptiveThreshold", "Lorg/opencv/core/Mat;", "image", "applyHeuristic", "", "probabilities", "", "calculateImageContrast", "calculateOmrBoxes", "Lorg/opencv/core/Rect;", "boxesDict", "multiplier", "imageToDraw", "calculateStandardDeviation", "values", "close", "", "detectBoxesInRoi", "roiMat", "minWidth", "maxWidth", "minHeight", "maxHeight", "roi", "drawPrediction", "prediction", "x", "y", "extractBoxesDict", "Lorg/informatika/sirekap/support/vision/Vision$BoxGroup;", "pageConfig", "Lorg/informatika/sirekap/model/FormConfig;", "threshImg", "widthScaleFactor", "heightScaleFactor", "getBoxesCoordinates", "croppedImage", "groupAndAddBoxesByField", "field", "Lorg/informatika/sirekap/model/FormConfig$Field;", "", "fieldBoxes", "boxType", "omrBoxes", "inferMissingCircles", "detectedBounds", "loadModelFileDescriptor", "Ljava/nio/ByteBuffer;", "modelPath", "loadModels", "locateBoxesInRegion", "nonMaxSuppression", "boxes", "overlapThresh", "predict", "correctedBitmap", "Landroid/graphics/Bitmap;", "formType", "electionType", "isLn", "", "isPos", "candidateNum", "maxCandidatesNum", "predictOcr", "boxGroups", "preprocessedImage", "numPaslon", "preprocessImage", "processField", "rWidth", "rHeight", "omrMultiplier", "processGroup", "boxGroup", "processOcr", "rect", "tempBoxGroup", "processOcrRegion", "region", "Lorg/informatika/sirekap/model/FormConfig$ROI;", "processOmr", "omrRect", "scanAreaOcr", "scanAreaOmr", "roiOmr", "grayCroppedImage", "selectCircle", "intensityValues", "BoxGroup", "Companion", "app_productionRelease"}, k = 1, mv = {1, 8, 0}, xi = 48)
/* compiled from: Vision.kt */
public final class Vision implements AutoCloseable {
    private static final double BOX_SIZE_TOLERANCE_OFFSET = 0.15d;
    private static final double CIRCLE_TO_CROPPED_WIDTH_RATIO = 0.016949152542373d;
    private static final Scalar COLOR_BLUE = new Scalar(255.0d, 0.0d, 0.0d);
    private static final Scalar COLOR_GREEN = new Scalar(0.0d, 255.0d, 0.0d);
    private static final Scalar COLOR_RED = new Scalar(0.0d, 0.0d, 255.0d);
    public static final Companion Companion = new Companion((DefaultConstructorMarker) null);
    private static final double OMR_BOX_SIZE_TOLERANCE_OFFSET = 0.7d;
    private static final double ROI_SIZE_NEGATIVE_OFFSET = 0.3d;
    private static final double ROI_SIZE_POSITIVE_OFFSET = 1.3d;
    private static final String TAG = "VISION";
    private String chosenMethodTabulation = "";
    private final Context context;
    private double fontScale = 1.0d;
    private List<Interpreter> interpreters;
    private Interpreter interpretersBlank;

    private final double adjustClipLimit(double d) {
        if (d < 50.0d) {
            return 3.0d;
        }
        return d < 100.0d ? 2.0d : 1.0d;
    }

    public Vision(Context context2) {
        Intrinsics.checkNotNullParameter(context2, "context");
        this.context = context2;
    }

    private final void loadModels() {
        ArrayList arrayList = new ArrayList(15);
        for (int i = 0; i < 15; i++) {
            arrayList.add("ensemble_model_" + i + ".tflite");
        }
        Iterable<String> iterable = arrayList;
        Collection arrayList2 = new ArrayList(CollectionsKt.collectionSizeOrDefault(iterable, 10));
        for (String str : iterable) {
            arrayList2.add("vision/ensemble-15-mnist/" + str);
        }
        Iterable<String> iterable2 = (List) arrayList2;
        Collection arrayList3 = new ArrayList(CollectionsKt.collectionSizeOrDefault(iterable2, 10));
        for (String loadModelFileDescriptor : iterable2) {
            arrayList3.add(new Interpreter(loadModelFileDescriptor(loadModelFileDescriptor), new Interpreter.Options()));
        }
        List<Interpreter> list = (List) arrayList3;
        this.interpreters = list;
        for (Interpreter allocateTensors : list) {
            allocateTensors.allocateTensors();
        }
        Interpreter interpreter = new Interpreter(loadModelFileDescriptor("vision/blank-dash-mult/blank_detection_model.tflite"), new Interpreter.Options());
        this.interpretersBlank = interpreter;
        interpreter.allocateTensors();
    }

    /* JADX WARNING: Code restructure failed: missing block: B:10:0x0044, code lost:
        throw r1;
     */
    /* JADX WARNING: Code restructure failed: missing block: B:8:0x0040, code lost:
        r1 = move-exception;
     */
    /* JADX WARNING: Code restructure failed: missing block: B:9:0x0041, code lost:
        kotlin.io.CloseableKt.closeFinally(r0, r6);
     */
    /* Code decompiled incorrectly, please refer to instructions dump. */
    private final java.nio.ByteBuffer loadModelFileDescriptor(java.lang.String r6) {
        /*
            r5 = this;
            android.content.Context r0 = r5.context
            android.content.res.AssetManager r0 = r0.getAssets()
            android.content.res.AssetFileDescriptor r6 = r0.openFd(r6)
            java.lang.String r0 = "assetManager.openFd(modelPath)"
            kotlin.jvm.internal.Intrinsics.checkNotNullExpressionValue(r6, r0)
            java.io.FileDescriptor r0 = r6.getFileDescriptor()
            long r1 = r6.getStartOffset()
            long r3 = r6.getDeclaredLength()
            int r6 = (int) r3
            java.nio.ByteBuffer r6 = java.nio.ByteBuffer.allocateDirect(r6)
            java.io.FileInputStream r3 = new java.io.FileInputStream
            r3.<init>(r0)
            java.nio.channels.FileChannel r0 = r3.getChannel()
            java.io.Closeable r0 = (java.io.Closeable) r0
            r3 = r0
            java.nio.channels.FileChannel r3 = (java.nio.channels.FileChannel) r3     // Catch:{ all -> 0x003e }
            r3.position(r1)     // Catch:{ all -> 0x003e }
            r3.read(r6)     // Catch:{ all -> 0x003e }
            r1 = 0
            kotlin.io.CloseableKt.closeFinally(r0, r1)
            java.lang.String r0 = "modelByteBuffer"
            kotlin.jvm.internal.Intrinsics.checkNotNullExpressionValue(r6, r0)
            return r6
        L_0x003e:
            r6 = move-exception
            throw r6     // Catch:{ all -> 0x0040 }
        L_0x0040:
            r1 = move-exception
            kotlin.io.CloseableKt.closeFinally(r0, r6)
            throw r1
        */
        throw new UnsupportedOperationException("Method not decompiled: org.informatika.sirekap.support.vision.Vision.loadModelFileDescriptor(java.lang.String):java.nio.ByteBuffer");
    }

    public final List<Integer> predict(Bitmap bitmap, String str, String str2, boolean z, boolean z2, int i, int i2) throws Exception {
        int i3;
        Bitmap bitmap2 = bitmap;
        String str3 = str;
        String str4 = str2;
        int i4 = i2;
        Intrinsics.checkNotNullParameter(bitmap2, "correctedBitmap");
        Intrinsics.checkNotNullParameter(str3, "formType");
        Intrinsics.checkNotNullParameter(str4, "electionType");
        if (!BuildConfig.IS_USE_LOCAL_OCR.booleanValue() && (!Intrinsics.areEqual((Object) str4, (Object) Election.ELECTION_PEMILIHAN_PRESIDEN) || !Intrinsics.areEqual((Object) str3, (Object) FormConfig.FORM_TALLY))) {
            return null;
        }
        if (OpenCVLoader.initDebug()) {
            loadModels();
            Mat mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC1);
            Utils.bitmapToMat(bitmap2, mat);
            this.fontScale = ((double) bitmap.getWidth()) / 2000.0d;
            Mat convertToGrayScale = ScanUtils.INSTANCE.convertToGrayScale(mat);
            Mat clone = mat.clone();
            Pair<Integer, Double> detect = ScanUtils.INSTANCE.detect(mat, bitmap2);
            if (detect != null) {
                i3 = AprilTagConfig.INSTANCE.getCandidateNum(detect.getFirst().intValue());
            } else if (Intrinsics.areEqual((Object) str4, (Object) Election.ELECTION_PEMILIHAN_DPD)) {
                boolean z3 = true;
                if ((1 <= i4 && i4 < 9) || (13 <= i4 && i4 < 17)) {
                    i3 = 8;
                } else {
                    if (!((((9 <= i4 && i4 < 11) || (17 <= i4 && i4 < 21)) || (25 <= i4 && i4 < 31)) || (37 <= i4 && i4 < 41)) && (49 > i4 || i4 >= 51)) {
                        z3 = false;
                    }
                    i3 = z3 ? 10 : 12;
                }
            } else {
                i3 = i;
            }
            Mat mat2 = clone;
            FormConfig config = FormConfig.Companion.getConfig(this.context, str, str2, z, z2, i3);
            Intrinsics.checkNotNullExpressionValue(mat2, "imageToDraw");
            List<BoxGroup> locateBoxesInRegion = locateBoxesInRegion(convertToGrayScale, config, mat2);
            Mat preprocessImage = preprocessImage(convertToGrayScale, locateBoxesInRegion);
            this.chosenMethodTabulation = "";
            Mat mat3 = mat2;
            Mat mat4 = convertToGrayScale;
            Mat mat5 = mat;
            Mat mat6 = preprocessImage;
            String str5 = "Vision Lokal (";
            String str6 = "_";
            predictOcr(locateBoxesInRegion, convertToGrayScale, mat6, i, mat3);
            Boolean bool = BuildConfig.IS_SAVE_VISION_ANNOTATION;
            Intrinsics.checkNotNullExpressionValue(bool, "IS_SAVE_VISION_ANNOTATION");
            if (bool.booleanValue()) {
                ScanUtils.INSTANCE.saveMatToGallery(this.context, mat3, str4 + str6 + str3 + str6 + (z ? "LN" : z2 ? "POS" : "DN") + str6 + (detect != null ? Integer.valueOf(detect.getFirst().intValue()) : "NONE"), str5 + this.chosenMethodTabulation + ")");
            }
            List<Integer> arrayList = new ArrayList<>();
            for (BoxGroup predictions : locateBoxesInRegion) {
                arrayList.add(Integer.valueOf(ElectionUtil.joinThreeNumbers(predictions.getPredictions())));
            }
            mat3.release();
            mat5.release();
            mat4.release();
            return arrayList;
        }
        throw new Exception("Unable to load OpenCV");
    }

    private final List<BoxGroup> locateBoxesInRegion(Mat mat, FormConfig formConfig, Mat mat2) {
        List<BoxGroup> boxesCoordinates = getBoxesCoordinates(mat, formConfig, mat2);
        for (BoxGroup next : boxesCoordinates) {
            int i = 0;
            for (Rect next2 : next.getCoordinates()) {
                int i2 = i + 1;
                if (i == 0) {
                    Imgproc.putText(mat2, next.getName(), new Point((double) next2.x, ((double) next2.y) - (Math.ceil(1.0d) * ((double) 12))), 0, 0.5d, COLOR_BLUE, (int) Math.ceil(1.0d));
                }
                i = i2;
            }
        }
        return boxesCoordinates;
    }

    private final Mat preprocessImage(Mat mat, List<BoxGroup> list) {
        Mat mat2 = new Mat(mat.size(), mat.type(), new Scalar(255.0d));
        for (BoxGroup coordinates : list) {
            for (Rect next : coordinates.getCoordinates()) {
                int i = next.x;
                int i2 = next.y;
                int i3 = next.width;
                int i4 = next.height;
                int min = Math.min((int) (((double) i3) * BOX_SIZE_TOLERANCE_OFFSET), i3 / 2);
                int min2 = Math.min((int) (((double) i4) * BOX_SIZE_TOLERANCE_OFFSET), i4 / 2);
                int i5 = i2 + min2;
                int i6 = (i2 + i4) - min2;
                int i7 = i + min;
                int i8 = (i + i3) - min;
                mat.submat(i5, i6, i7, i8).copyTo(mat2.submat(i5, i6, i7, i8));
            }
        }
        return mat2;
    }

    private final void predictOcr(List<BoxGroup> list, Mat mat, Mat mat2, int i, Mat mat3) {
        Mat mat4 = mat;
        Mat mat5 = mat2;
        int i2 = i;
        Mat mat6 = mat3;
        for (BoxGroup next : list) {
            if (StringsKt.startsWith$default(next.getName(), "suara_paslon_", false, 2, (Object) null)) {
                Integer intOrNull = StringsKt.toIntOrNull((String) StringsKt.split$default((CharSequence) next.getName(), new String[]{"suara_paslon_"}, false, 0, 6, (Object) null).get(1));
                if (intOrNull == null || intOrNull.intValue() <= i2 || i2 == 0) {
                    processGroup(next, mat4, mat5, mat6);
                }
            } else {
                processGroup(next, mat4, mat5, mat6);
            }
        }
    }

    private final void processGroup(BoxGroup boxGroup, Mat mat, Mat mat2, Mat mat3) {
        BoxGroup boxGroup2 = boxGroup;
        BoxGroup copy$default = BoxGroup.copy$default(boxGroup, (String) null, (List) null, (String) null, (List) null, 15, (Object) null);
        for (Rect processOcr : boxGroup.getCoordinates()) {
            processOcr(boxGroup, processOcr, mat2, mat3, copy$default);
        }
        if (Intrinsics.areEqual((Object) boxGroup.getType(), (Object) "omr")) {
            try {
                for (Rect processOmr : boxGroup.getCoordinatesOmr()) {
                    processOmr(boxGroup2, processOmr, mat, mat3);
                }
                this.chosenMethodTabulation += "OMR,";
            } catch (Exception unused) {
                boxGroup.deletePredictions();
                for (Integer intValue : copy$default.getPredictions()) {
                    boxGroup2.addPrediction(intValue.intValue());
                }
                this.chosenMethodTabulation += "OCR,";
            }
        }
    }

    private final void processOcr(BoxGroup boxGroup, Rect rect, Mat mat, Mat mat2, BoxGroup boxGroup2) {
        int coerceAtMost = RangesKt.coerceAtMost((int) (((double) rect.width) * BOX_SIZE_TOLERANCE_OFFSET), rect.width / 2);
        int coerceAtMost2 = RangesKt.coerceAtMost((int) (((double) rect.height) * BOX_SIZE_TOLERANCE_OFFSET), rect.height / 2);
        Mat submat = mat.submat(rect.y + coerceAtMost2, (rect.y + rect.height) - coerceAtMost2, rect.x + coerceAtMost, (rect.x + rect.width) - coerceAtMost);
        Intrinsics.checkNotNullExpressionValue(submat, "roi");
        int scanAreaOcr = scanAreaOcr(submat);
        if (Intrinsics.areEqual((Object) boxGroup.getType(), (Object) "ocr")) {
            boxGroup.addPrediction(scanAreaOcr);
        } else {
            boxGroup2.addPrediction(scanAreaOcr);
        }
        drawPrediction(scanAreaOcr, rect.x, rect.y, mat2);
    }

    private final int scanAreaOcr(Mat mat) {
        Mat clone = mat.clone();
        Mat mat2 = new Mat();
        Imgproc.resize(clone, mat2, new Size(28.0d, 28.0d));
        Imgproc.adaptiveThreshold(mat2, mat2, 255.0d, 1, 0, 11, 7.0d);
        Mat mat3 = new Mat();
        Core.bitwise_not(mat2, mat3);
        ByteBuffer allocateDirect = ByteBuffer.allocateDirect(mat3.width() * 4 * mat3.height());
        allocateDirect.order(ByteOrder.nativeOrder());
        int height = mat3.height();
        for (int i = 0; i < height; i++) {
            int width = mat3.width();
            for (int i2 = 0; i2 < width; i2++) {
                allocateDirect.putFloat((float) (mat3.get(i, i2)[0] / ((double) 255.0f)));
            }
        }
        Interpreter interpreter = this.interpretersBlank;
        List<Interpreter> list = null;
        if (interpreter == null) {
            Intrinsics.throwUninitializedPropertyAccessException("interpretersBlank");
            interpreter = null;
        }
        Tensor inputTensor = interpreter.getInputTensor(0);
        Interpreter interpreter2 = this.interpretersBlank;
        if (interpreter2 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("interpretersBlank");
            interpreter2 = null;
        }
        Tensor outputTensor = interpreter2.getOutputTensor(0);
        TensorBuffer createFixedSize = TensorBuffer.createFixedSize(inputTensor.shape(), inputTensor.dataType());
        Intrinsics.checkNotNullExpressionValue(createFixedSize, "createFixedSize(blankInp…kInputDetails.dataType())");
        createFixedSize.loadBuffer(allocateDirect);
        TensorBuffer createFixedSize2 = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType());
        Intrinsics.checkNotNullExpressionValue(createFixedSize2, "createFixedSize(blankOut…OutputDetails.dataType())");
        Interpreter interpreter3 = this.interpretersBlank;
        if (interpreter3 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("interpretersBlank");
            interpreter3 = null;
        }
        interpreter3.run(createFixedSize.getBuffer(), createFixedSize2.getBuffer());
        float[] floatArray = createFixedSize2.getFloatArray();
        Intrinsics.checkNotNullExpressionValue(floatArray, "blankOutputData.floatArray");
        Float maxOrNull = ArraysKt.maxOrNull(floatArray);
        if ((maxOrNull != null ? maxOrNull.floatValue() : 0.0f) >= 1.0f) {
            return 0;
        }
        float[] fArr = new float[10];
        List<Interpreter> list2 = this.interpreters;
        if (list2 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("interpreters");
        } else {
            list = list2;
        }
        for (Interpreter next : list) {
            Tensor inputTensor2 = next.getInputTensor(0);
            Tensor outputTensor2 = next.getOutputTensor(0);
            TensorBuffer createFixedSize3 = TensorBuffer.createFixedSize(inputTensor2.shape(), inputTensor2.dataType());
            Intrinsics.checkNotNullExpressionValue(createFixedSize3, "createFixedSize(inputDet… inputDetails.dataType())");
            createFixedSize3.loadBuffer(allocateDirect);
            TensorBuffer createFixedSize4 = TensorBuffer.createFixedSize(outputTensor2.shape(), outputTensor2.dataType());
            Intrinsics.checkNotNullExpressionValue(createFixedSize4, "createFixedSize(outputDe…outputDetails.dataType())");
            next.run(createFixedSize3.getBuffer(), createFixedSize4.getBuffer());
            for (int i3 = 0; i3 < 10; i3++) {
                fArr[i3] = fArr[i3] + createFixedSize4.getFloatArray()[i3];
            }
        }
        return applyHeuristic(fArr);
    }

    /* JADX DEBUG: Multi-variable search result rejected for TypeSearchVarInfo{r2v8, resolved type: java.lang.Number} */
    /* JADX WARNING: Multi-variable type inference failed */
    /* Code decompiled incorrectly, please refer to instructions dump. */
    private final int applyHeuristic(float[] r11) {
        /*
            r10 = this;
            kotlin.ranges.IntRange r0 = kotlin.collections.ArraysKt.getIndices((float[]) r11)
            java.lang.Iterable r0 = (java.lang.Iterable) r0
            java.util.Iterator r0 = r0.iterator()
            boolean r1 = r0.hasNext()
            if (r1 != 0) goto L_0x0012
            r0 = 0
            goto L_0x0043
        L_0x0012:
            java.lang.Object r1 = r0.next()
            boolean r2 = r0.hasNext()
            if (r2 != 0) goto L_0x001e
        L_0x001c:
            r0 = r1
            goto L_0x0043
        L_0x001e:
            r2 = r1
            java.lang.Number r2 = (java.lang.Number) r2
            int r2 = r2.intValue()
            r2 = r11[r2]
        L_0x0027:
            java.lang.Object r3 = r0.next()
            r4 = r3
            java.lang.Number r4 = (java.lang.Number) r4
            int r4 = r4.intValue()
            r4 = r11[r4]
            int r5 = java.lang.Float.compare(r2, r4)
            if (r5 >= 0) goto L_0x003c
            r1 = r3
            r2 = r4
        L_0x003c:
            boolean r3 = r0.hasNext()
            if (r3 != 0) goto L_0x0027
            goto L_0x001c
        L_0x0043:
            java.lang.Integer r0 = (java.lang.Integer) r0
            r1 = 0
            if (r0 == 0) goto L_0x004d
            int r0 = r0.intValue()
            goto L_0x004e
        L_0x004d:
            r0 = r1
        L_0x004e:
            kotlin.ranges.IntRange r2 = kotlin.collections.ArraysKt.getIndices((float[]) r11)
            java.lang.Iterable r2 = (java.lang.Iterable) r2
            org.informatika.sirekap.support.vision.Vision$applyHeuristic$$inlined$sortedByDescending$1 r3 = new org.informatika.sirekap.support.vision.Vision$applyHeuristic$$inlined$sortedByDescending$1
            r3.<init>(r11)
            java.util.Comparator r3 = (java.util.Comparator) r3
            java.util.List r2 = kotlin.collections.CollectionsKt.sortedWith(r2, r3)
            r3 = 1
            java.lang.Object r2 = r2.get(r3)
            java.lang.Number r2 = (java.lang.Number) r2
            int r2 = r2.intValue()
            r4 = 4621819117588971520(0x4024000000000000, double:10.0)
            if (r0 != 0) goto L_0x007a
            r1 = r11[r1]
            double r6 = (double) r1
            int r1 = (r6 > r4 ? 1 : (r6 == r4 ? 0 : -1))
            if (r1 >= 0) goto L_0x007a
            r11 = 8
            if (r2 != r11) goto L_0x00a5
            return r2
        L_0x007a:
            if (r0 != r3) goto L_0x008f
            r1 = r11[r3]
            double r6 = (double) r1
            r8 = 4624352392379367424(0x402d000000000000, double:14.5)
            int r1 = (r6 > r8 ? 1 : (r6 == r8 ? 0 : -1))
            if (r1 >= 0) goto L_0x008f
            r11 = r11[r2]
            double r3 = (double) r11
            r5 = 0
            int r11 = (r3 > r5 ? 1 : (r3 == r5 ? 0 : -1))
            if (r11 <= 0) goto L_0x00a5
            return r2
        L_0x008f:
            r1 = 3
            if (r0 != r1) goto L_0x00a5
            r1 = r11[r1]
            double r1 = (double) r1
            int r1 = (r1 > r4 ? 1 : (r1 == r4 ? 0 : -1))
            if (r1 >= 0) goto L_0x00a5
            r1 = 9
            r11 = r11[r1]
            double r2 = (double) r11
            r4 = 4613937818241073152(0x4008000000000000, double:3.0)
            int r11 = (r2 > r4 ? 1 : (r2 == r4 ? 0 : -1))
            if (r11 <= 0) goto L_0x00a5
            return r1
        L_0x00a5:
            return r0
        */
        throw new UnsupportedOperationException("Method not decompiled: org.informatika.sirekap.support.vision.Vision.applyHeuristic(float[]):int");
    }

    private final void processOmr(BoxGroup boxGroup, Rect rect, Mat mat, Mat mat2) {
        Mat submat = mat.submat(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width);
        Intrinsics.checkNotNullExpressionValue(submat, "roiOmr");
        int scanAreaOmr = scanAreaOmr(submat, mat, mat2, rect);
        boxGroup.addPrediction(scanAreaOmr);
        drawPrediction(scanAreaOmr, rect.x, rect.y, mat2);
    }

    private final void drawPrediction(int i, int i2, int i3, Mat mat) {
        double d = this.fontScale;
        Mat mat2 = mat;
        Imgproc.putText(mat2, String.valueOf(i), new Point(((double) i2) + (((double) 7) * d), ((double) i3) + (d * ((double) 37))), 0, 0.5d, COLOR_BLUE, 1);
    }

    private final int scanAreaOmr(Mat mat, Mat mat2, Mat mat3, Rect rect) {
        Mat mat4 = mat3;
        Rect rect2 = rect;
        Mat clone = mat.clone();
        int i = rect2.x;
        int i2 = rect2.y;
        int ceil = (int) Math.ceil(((double) mat.width()) / 63.5d);
        int max = (int) Math.max(((double) Math.min(clone.width(), clone.height())) * 0.4d, 3.0d);
        if (max % 2 == 0) {
            max++;
        }
        int i3 = max;
        Mat mat5 = new Mat();
        Mat mat6 = mat5;
        Imgproc.adaptiveThreshold(clone, mat5, 255.0d, 1, 1, i3, (double) ((int) (((double) i3) * 0.2d)));
        List arrayList = new ArrayList();
        Imgproc.findContours(mat6, arrayList, new Mat(), 3, 2);
        double d = mat2.size().width * CIRCLE_TO_CROPPED_WIDTH_RATIO;
        double pow = Math.pow(d * 0.5d, 2.0d) * 3.141592653589793d;
        double d2 = pow * 0.30000000000000004d;
        double d3 = pow * 1.7d;
        double d4 = 0.30000000000000004d * d;
        double d5 = d * 1.7d;
        List arrayList2 = new ArrayList();
        Iterator it = arrayList.iterator();
        while (it.hasNext()) {
            Rect boundingRect = Imgproc.boundingRect((MatOfPoint) it.next());
            int i4 = boundingRect.width;
            int i5 = boundingRect.height;
            Iterator it2 = it;
            int i6 = i2;
            int i7 = ceil;
            double pow2 = Math.pow((((double) (i4 + i5)) / 2.0d) * 0.5d, 2.0d) * 3.141592653589793d;
            if (d2 < pow2 && pow2 < d3) {
                double d6 = (double) i4;
                if (d4 <= d6 && d6 <= d5) {
                    double d7 = (double) i5;
                    if (d4 <= d7 && d7 <= d5) {
                        Intrinsics.checkNotNullExpressionValue(boundingRect, "boundingRect");
                        arrayList2.add(boundingRect);
                    }
                }
            }
            it = it2;
            i2 = i6;
            ceil = i7;
        }
        int i8 = i2;
        int i9 = ceil;
        List<Rect> nonMaxSuppression = nonMaxSuppression(arrayList2, 0.2d);
        Iterable iterable = nonMaxSuppression;
        CollectionsKt.sortedWith(iterable, new Vision$scanAreaOmr$$inlined$sortedBy$1());
        if (nonMaxSuppression.size() >= 10) {
            List take = CollectionsKt.take(iterable, 10);
            List arrayList3 = new ArrayList();
            int size = take.size();
            int i10 = 0;
            while (i10 < size) {
                int i11 = ((Rect) take.get(i10)).x;
                int i12 = ((Rect) take.get(i10)).y;
                int i13 = ((Rect) take.get(i10)).width;
                int i14 = ((Rect) take.get(i10)).height;
                Mat submat = mat6.submat(i12, i12 + i14, i11, i11 + i13);
                int i15 = i11 + i;
                int i16 = i12 + i8;
                Imgproc.rectangle(mat3, new Point((double) i15, (double) i16), new Point((double) (i15 + i13), (double) (i16 + i14)), COLOR_GREEN, i9);
                arrayList3.add(Double.valueOf(Core.mean(submat).val[0]));
                nonMaxSuppression = nonMaxSuppression;
                i10++;
                mat6 = mat6;
            }
            List<Rect> list = nonMaxSuppression;
            int selectCircle = selectCircle(arrayList3);
            Rect rect3 = nonMaxSuppression.get(selectCircle);
            int i17 = rect3.x;
            int i18 = rect3.y;
            int i19 = rect3.width;
            int i20 = rect3.height;
            int i21 = i17 + i;
            int i22 = i18 + i8;
            Imgproc.rectangle(mat3, new Point((double) i21, (double) i22), new Point((double) (i21 + i19), (double) (i22 + i20)), COLOR_RED, i9);
            return 9 - selectCircle;
        }
        throw new IllegalArgumentException("Insufficient data to perform OMR circle inference. Detected count: " + nonMaxSuppression.size());
    }

    private final int selectCircle(List<Double> list) {
        if (list.isEmpty()) {
            return 0;
        }
        double averageOfDouble = CollectionsKt.averageOfDouble(list);
        double calculateStandardDeviation = calculateStandardDeviation(list) + averageOfDouble;
        double d = averageOfDouble * 0.2d;
        int size = list.size();
        for (int i = 0; i < size; i++) {
            double doubleValue = list.get(i).doubleValue();
            if (doubleValue > calculateStandardDeviation) {
                if (i == list.size() - 1) {
                    if (Math.abs(list.get(i - 1).doubleValue() - doubleValue) > d) {
                    }
                } else if (Math.abs(list.get(i + 1).doubleValue() - doubleValue) > d) {
                }
                return i;
            }
        }
        return 0;
    }

    private final double calculateStandardDeviation(List<Double> list) {
        Iterable<Number> iterable = list;
        double averageOfDouble = CollectionsKt.averageOfDouble(iterable);
        Collection arrayList = new ArrayList(CollectionsKt.collectionSizeOrDefault(iterable, 10));
        for (Number doubleValue : iterable) {
            arrayList.add(Double.valueOf(Math.pow(doubleValue.doubleValue() - averageOfDouble, 2.0d)));
        }
        return Math.sqrt(CollectionsKt.averageOfDouble((List) arrayList));
    }

    private final List<Rect> inferMissingCircles(List<? extends Rect> list) {
        List<? extends Rect> list2 = list;
        List<Rect> arrayList = new ArrayList<>();
        Iterable until = RangesKt.until(0, list.size() - 1);
        Collection arrayList2 = new ArrayList(CollectionsKt.collectionSizeOrDefault(until, 10));
        Iterator it = until.iterator();
        while (it.hasNext()) {
            int nextInt = ((IntIterator) it).nextInt();
            arrayList2.add(Integer.valueOf(((Rect) list2.get(nextInt + 1)).y - ((Rect) list2.get(nextInt)).y));
        }
        List list3 = (List) arrayList2;
        int size = list3.size();
        int i = 0;
        while (i < size) {
            int i2 = i + 1;
            int size2 = list3.size();
            int i3 = i2;
            while (i3 < size2) {
                int i4 = i3 + 1;
                List subList = list3.subList(i, i4);
                if (inferMissingCircles$isConsistent(0.2d, subList)) {
                    double sumOfInt = ((double) CollectionsKt.sumOfInt(subList)) / ((double) subList.size());
                    if (1 <= i) {
                        int i5 = 1;
                        while (true) {
                            arrayList.add(0, new Rect(((Rect) list2.get(i)).x, (int) (((double) ((Rect) list2.get(i)).y) - (((double) i5) * sumOfInt)), ((Rect) list2.get(i)).width, ((Rect) list2.get(i)).height));
                            if (i5 == i) {
                                break;
                            }
                            i5++;
                        }
                    }
                    int size3 = list3.size() - i3;
                    for (int i6 = 1; i6 < size3; i6++) {
                        arrayList.add(new Rect(((Rect) list2.get(i3)).x, (int) (((double) ((Rect) list2.get(i3)).y) + (((double) i6) * sumOfInt)), ((Rect) list2.get(i3)).width, ((Rect) list2.get(i3)).height));
                    }
                    return arrayList;
                }
                i3 = i4;
            }
            i = i2;
        }
        return arrayList;
    }

    private static final boolean inferMissingCircles$isConsistent(double d, List<Integer> list) {
        boolean z;
        if (list.size() < 2) {
            return false;
        }
        Iterable<Number> iterable = list;
        double averageOfInt = CollectionsKt.averageOfInt(iterable);
        if (!(iterable instanceof Collection) || !((Collection) iterable).isEmpty()) {
            for (Number intValue : iterable) {
                if (Math.abs(((double) intValue.intValue()) - averageOfInt) <= d * averageOfInt) {
                    z = true;
                    continue;
                } else {
                    z = false;
                    continue;
                }
                if (!z) {
                    return false;
                }
            }
        }
        return true;
    }

    private final List<BoxGroup> getBoxesCoordinates(Mat mat, FormConfig formConfig, Mat mat2) {
        int width = mat.width();
        int width2 = formConfig.getWidth();
        int height = mat.height();
        return extractBoxesDict(formConfig, applyAdaptiveThreshold(mat), ((double) width) / ((double) width2), ((double) height) / ((double) formConfig.getHeight()), mat2);
    }

    private final List<BoxGroup> extractBoxesDict(FormConfig formConfig, Mat mat, double d, double d2, Mat mat2) {
        List<BoxGroup> arrayList = new ArrayList<>();
        for (FormConfig.ROI next : formConfig.getRegionOfInterest()) {
            processOcrRegion(next, arrayList, mat, d, d2, next.getType(), mat2);
        }
        return arrayList;
    }

    private final void processOcrRegion(FormConfig.ROI roi, List<BoxGroup> list, Mat mat, double d, double d2, String str, Mat mat2) {
        double height = ((double) roi.getHeight()) * d2;
        double width = ((double) roi.getWidth()) * d;
        double d3 = (double) 1;
        double d4 = d3 - BOX_SIZE_TOLERANCE_OFFSET;
        double d5 = d4 * width;
        double d6 = d3 + BOX_SIZE_TOLERANCE_OFFSET;
        double d7 = d6 * width;
        double d8 = d4 * height;
        double d9 = d6 * height;
        Double omrBoundMultiplier = roi.getOmrBoundMultiplier();
        double doubleValue = omrBoundMultiplier != null ? omrBoundMultiplier.doubleValue() : 0.0d;
        Mat clone = mat.clone();
        for (FormConfig.Field processField : roi.getFields()) {
            Intrinsics.checkNotNullExpressionValue(clone, "combinedLines");
            int i = (int) height;
            double d10 = height;
            double d11 = d9;
            Mat mat3 = clone;
            processField(processField, list, mat3, (int) width, i, (int) d5, (int) d7, (int) d8, (int) d11, d, d2, str, doubleValue, mat2);
            clone = mat3;
            height = d10;
            d8 = d8;
            d7 = d7;
            d5 = d5;
            width = width;
            d9 = d11;
        }
    }

    private final void processField(FormConfig.Field field, List<BoxGroup> list, Mat mat, int i, int i2, int i3, int i4, int i5, int i6, double d, double d2, String str, double d3, Mat mat2) {
        int i7 = i;
        int i8 = i2;
        Mat mat3 = mat2;
        List<List<Integer>> coordinates = field.getCoordinates();
        List arrayList = new ArrayList();
        for (List next : coordinates) {
            int doubleValue = (int) (((Number) next.get(0)).doubleValue() * d);
            int doubleValue2 = (int) (((Number) next.get(1)).doubleValue() * d2);
            double d4 = (double) i7;
            int max = Math.max(0, doubleValue - ((int) (d4 * ROI_SIZE_NEGATIVE_OFFSET)));
            double d5 = (double) i8;
            int max2 = Math.max(0, doubleValue2 - ((int) (ROI_SIZE_NEGATIVE_OFFSET * d5)));
            Rect rect = new Rect(max, max2, Math.min(mat.width(), ((int) (d4 * ROI_SIZE_POSITIVE_OFFSET)) + doubleValue) - max, Math.min(mat.height(), ((int) (d5 * ROI_SIZE_POSITIVE_OFFSET)) + doubleValue2) - max2);
            List<Rect> detectBoxesInRoi = detectBoxesInRoi(new Mat(mat, rect), i3, i4, i5, i6, rect);
            if (detectBoxesInRoi.isEmpty()) {
                Rect rect2 = new Rect(doubleValue, doubleValue2, i7, i8);
                arrayList.add(rect2);
                Imgproc.rectangle(mat3, new Point((double) rect2.x, (double) rect2.y), new Point((double) (rect2.x + rect2.width), (double) (rect2.y + rect2.height)), COLOR_RED, 1);
            } else {
                arrayList.addAll(detectBoxesInRoi);
                Imgproc.rectangle(mat3, new Point((double) detectBoxesInRoi.get(0).x, (double) detectBoxesInRoi.get(0).y), new Point((double) (detectBoxesInRoi.get(0).x + detectBoxesInRoi.get(0).width), (double) (detectBoxesInRoi.get(0).y + detectBoxesInRoi.get(0).height)), COLOR_BLUE, 1);
            }
        }
        if (Intrinsics.areEqual((Object) str, (Object) "ocr")) {
            groupAndAddBoxesByField$default(this, field, list, arrayList, str, (List) null, 16, (Object) null);
            return;
        }
        groupAndAddBoxesByField(field, list, arrayList, str, calculateOmrBoxes(arrayList, d3, mat3));
    }

    private final List<Rect> calculateOmrBoxes(List<? extends Rect> list, double d, Mat mat) {
        List<Rect> arrayList = new ArrayList<>();
        for (Rect rect : list) {
            int i = rect.x;
            int i2 = rect.y;
            int i3 = rect.width;
            int i4 = rect.height;
            Rect rect2 = new Rect(i, i2 + i4, i3, (int) (((double) i4) * d));
            arrayList.add(rect2);
            Imgproc.rectangle(mat, rect2.tl(), rect2.br(), COLOR_BLUE, 1);
        }
        return arrayList;
    }

    private final List<Rect> detectBoxesInRoi(Mat mat, int i, int i2, int i3, int i4, Rect rect) {
        List<MatOfPoint> arrayList = new ArrayList<>();
        Imgproc.findContours(mat, arrayList, new Mat(), 3, 2);
        List arrayList2 = new ArrayList();
        for (MatOfPoint boundingRect : arrayList) {
            Rect boundingRect2 = Imgproc.boundingRect(boundingRect);
            int i5 = boundingRect2.width;
            boolean z = false;
            if (i <= i5 && i5 <= i2) {
                z = true;
            }
            if (z && i3 <= boundingRect2.height && boundingRect2.height <= i4) {
                arrayList2.add(new Rect(boundingRect2.x + rect.x, boundingRect2.y + rect.y, boundingRect2.width, boundingRect2.height));
            }
        }
        return nonMaxSuppression(arrayList2, ROI_SIZE_NEGATIVE_OFFSET);
    }

    private final List<Rect> nonMaxSuppression(List<? extends Rect> list, double d) {
        List list2;
        List<? extends Rect> list3 = list;
        if (list.isEmpty()) {
            return new ArrayList<>();
        }
        Iterable<Rect> iterable = list3;
        Collection arrayList = new ArrayList(CollectionsKt.collectionSizeOrDefault(iterable, 10));
        for (Rect rect : iterable) {
            arrayList.add(Double.valueOf((double) rect.x));
        }
        double[] doubleArray = CollectionsKt.toDoubleArray((List) arrayList);
        Collection arrayList2 = new ArrayList(CollectionsKt.collectionSizeOrDefault(iterable, 10));
        for (Rect rect2 : iterable) {
            arrayList2.add(Double.valueOf((double) rect2.y));
        }
        double[] doubleArray2 = CollectionsKt.toDoubleArray((List) arrayList2);
        Collection arrayList3 = new ArrayList(CollectionsKt.collectionSizeOrDefault(iterable, 10));
        for (Rect rect3 : iterable) {
            arrayList3.add(Double.valueOf((double) (rect3.x + rect3.width)));
        }
        double[] doubleArray3 = CollectionsKt.toDoubleArray((List) arrayList3);
        Collection arrayList4 = new ArrayList(CollectionsKt.collectionSizeOrDefault(iterable, 10));
        for (Rect rect4 : iterable) {
            arrayList4.add(Double.valueOf((double) (rect4.y + rect4.height)));
        }
        double[] doubleArray4 = CollectionsKt.toDoubleArray((List) arrayList4);
        int[] intArray = CollectionsKt.toIntArray(CollectionsKt.sortedWith(CollectionsKt.getIndices(list3), new Vision$nonMaxSuppression$$inlined$sortedBy$1(doubleArray4)));
        List<Rect> arrayList5 = new ArrayList<>();
        List<Integer> mutableList = ArraysKt.toMutableList(intArray);
        while (!mutableList.isEmpty()) {
            int size = mutableList.size() - 1;
            int intValue = mutableList.get(size).intValue();
            arrayList5.add(list3.get(intValue));
            int i = 0;
            List mutableListOf = CollectionsKt.mutableListOf(Integer.valueOf(size));
            int i2 = 0;
            while (i2 < size) {
                int intValue2 = mutableList.get(i2).intValue();
                List<Rect> list4 = arrayList5;
                int i3 = size;
                List list5 = mutableListOf;
                int i4 = i2;
                List<Integer> list6 = mutableList;
                double[] dArr = doubleArray2;
                int i5 = intValue;
                if ((Math.max(0.0d, Math.min(doubleArray3[intValue], doubleArray3[intValue2]) - Math.max(doubleArray[intValue], doubleArray[intValue2])) * Math.max(0.0d, Math.min(doubleArray4[intValue], doubleArray4[intValue2]) - Math.max(doubleArray2[intValue], doubleArray2[intValue2]))) / ((doubleArray3[intValue2] - doubleArray[intValue2]) * (doubleArray4[intValue2] - dArr[intValue2])) > d) {
                    list2 = list5;
                    list2.add(Integer.valueOf(i4));
                } else {
                    list2 = list5;
                }
                i2 = i4 + 1;
                mutableListOf = list2;
                arrayList5 = list4;
                size = i3;
                doubleArray2 = dArr;
                mutableList = list6;
                intValue = i5;
            }
            double[] dArr2 = doubleArray2;
            List<Rect> list7 = arrayList5;
            List list8 = mutableListOf;
            Collection arrayList6 = new ArrayList();
            for (Object next : mutableList) {
                int i6 = i + 1;
                if (i < 0) {
                    CollectionsKt.throwIndexOverflow();
                }
                ((Number) next).intValue();
                if (!list8.contains(Integer.valueOf(i))) {
                    arrayList6.add(next);
                }
                i = i6;
            }
            mutableList = CollectionsKt.toMutableList((List) arrayList6);
            arrayList5 = list7;
            doubleArray2 = dArr2;
        }
        return arrayList5;
    }

    @Metadata(d1 = {"\u00008\n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0000\n\u0002\u0010\u000e\n\u0000\n\u0002\u0010 \n\u0002\u0018\u0002\n\u0002\b\t\n\u0002\u0010!\n\u0002\u0010\b\n\u0002\b\u0003\n\u0002\u0010\u0002\n\u0002\b\b\n\u0002\u0010\u000b\n\u0002\b\u0004\b\b\u0018\u00002\u00020\u0001B1\u0012\u0006\u0010\u0002\u001a\u00020\u0003\u0012\f\u0010\u0004\u001a\b\u0012\u0004\u0012\u00020\u00060\u0005\u0012\u0006\u0010\u0007\u001a\u00020\u0003\u0012\f\u0010\b\u001a\b\u0012\u0004\u0012\u00020\u00060\u0005¢\u0006\u0002\u0010\tJ\u000e\u0010\u0014\u001a\u00020\u00152\u0006\u0010\u0016\u001a\u00020\u0011J\t\u0010\u0017\u001a\u00020\u0003HÆ\u0003J\u000f\u0010\u0018\u001a\b\u0012\u0004\u0012\u00020\u00060\u0005HÆ\u0003J\t\u0010\u0019\u001a\u00020\u0003HÆ\u0003J\u000f\u0010\u001a\u001a\b\u0012\u0004\u0012\u00020\u00060\u0005HÆ\u0003J=\u0010\u001b\u001a\u00020\u00002\b\b\u0002\u0010\u0002\u001a\u00020\u00032\u000e\b\u0002\u0010\u0004\u001a\b\u0012\u0004\u0012\u00020\u00060\u00052\b\b\u0002\u0010\u0007\u001a\u00020\u00032\u000e\b\u0002\u0010\b\u001a\b\u0012\u0004\u0012\u00020\u00060\u0005HÆ\u0001J\u0006\u0010\u001c\u001a\u00020\u0015J\u0013\u0010\u001d\u001a\u00020\u001e2\b\u0010\u001f\u001a\u0004\u0018\u00010\u0001HÖ\u0003J\t\u0010 \u001a\u00020\u0011HÖ\u0001J\t\u0010!\u001a\u00020\u0003HÖ\u0001R\u0017\u0010\u0004\u001a\b\u0012\u0004\u0012\u00020\u00060\u0005¢\u0006\b\n\u0000\u001a\u0004\b\n\u0010\u000bR\u0017\u0010\b\u001a\b\u0012\u0004\u0012\u00020\u00060\u0005¢\u0006\b\n\u0000\u001a\u0004\b\f\u0010\u000bR\u0011\u0010\u0002\u001a\u00020\u0003¢\u0006\b\n\u0000\u001a\u0004\b\r\u0010\u000eR\u0017\u0010\u000f\u001a\b\u0012\u0004\u0012\u00020\u00110\u0010¢\u0006\b\n\u0000\u001a\u0004\b\u0012\u0010\u000bR\u0011\u0010\u0007\u001a\u00020\u0003¢\u0006\b\n\u0000\u001a\u0004\b\u0013\u0010\u000e¨\u0006\""}, d2 = {"Lorg/informatika/sirekap/support/vision/Vision$BoxGroup;", "", "name", "", "coordinates", "", "Lorg/opencv/core/Rect;", "type", "coordinatesOmr", "(Ljava/lang/String;Ljava/util/List;Ljava/lang/String;Ljava/util/List;)V", "getCoordinates", "()Ljava/util/List;", "getCoordinatesOmr", "getName", "()Ljava/lang/String;", "predictions", "", "", "getPredictions", "getType", "addPrediction", "", "prediction", "component1", "component2", "component3", "component4", "copy", "deletePredictions", "equals", "", "other", "hashCode", "toString", "app_productionRelease"}, k = 1, mv = {1, 8, 0}, xi = 48)
    /* compiled from: Vision.kt */
    public static final class BoxGroup {
        private final List<Rect> coordinates;
        private final List<Rect> coordinatesOmr;
        private final String name;
        private final List<Integer> predictions = new ArrayList();
        private final String type;

        public static /* synthetic */ BoxGroup copy$default(BoxGroup boxGroup, String str, List<Rect> list, String str2, List<Rect> list2, int i, Object obj) {
            if ((i & 1) != 0) {
                str = boxGroup.name;
            }
            if ((i & 2) != 0) {
                list = boxGroup.coordinates;
            }
            if ((i & 4) != 0) {
                str2 = boxGroup.type;
            }
            if ((i & 8) != 0) {
                list2 = boxGroup.coordinatesOmr;
            }
            return boxGroup.copy(str, list, str2, list2);
        }

        public final String component1() {
            return this.name;
        }

        public final List<Rect> component2() {
            return this.coordinates;
        }

        public final String component3() {
            return this.type;
        }

        public final List<Rect> component4() {
            return this.coordinatesOmr;
        }

        public final BoxGroup copy(String str, List<? extends Rect> list, String str2, List<? extends Rect> list2) {
            Intrinsics.checkNotNullParameter(str, "name");
            Intrinsics.checkNotNullParameter(list, "coordinates");
            Intrinsics.checkNotNullParameter(str2, "type");
            Intrinsics.checkNotNullParameter(list2, "coordinatesOmr");
            return new BoxGroup(str, list, str2, list2);
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof BoxGroup)) {
                return false;
            }
            BoxGroup boxGroup = (BoxGroup) obj;
            return Intrinsics.areEqual((Object) this.name, (Object) boxGroup.name) && Intrinsics.areEqual((Object) this.coordinates, (Object) boxGroup.coordinates) && Intrinsics.areEqual((Object) this.type, (Object) boxGroup.type) && Intrinsics.areEqual((Object) this.coordinatesOmr, (Object) boxGroup.coordinatesOmr);
        }

        public int hashCode() {
            return (((((this.name.hashCode() * 31) + this.coordinates.hashCode()) * 31) + this.type.hashCode()) * 31) + this.coordinatesOmr.hashCode();
        }

        public String toString() {
            String str = this.name;
            List<Rect> list = this.coordinates;
            String str2 = this.type;
            return "BoxGroup(name=" + str + ", coordinates=" + list + ", type=" + str2 + ", coordinatesOmr=" + this.coordinatesOmr + ")";
        }

        public BoxGroup(String str, List<? extends Rect> list, String str2, List<? extends Rect> list2) {
            Intrinsics.checkNotNullParameter(str, "name");
            Intrinsics.checkNotNullParameter(list, "coordinates");
            Intrinsics.checkNotNullParameter(str2, "type");
            Intrinsics.checkNotNullParameter(list2, "coordinatesOmr");
            this.name = str;
            this.coordinates = list;
            this.type = str2;
            this.coordinatesOmr = list2;
        }

        public final String getName() {
            return this.name;
        }

        public final List<Rect> getCoordinates() {
            return this.coordinates;
        }

        public final String getType() {
            return this.type;
        }

        public final List<Rect> getCoordinatesOmr() {
            return this.coordinatesOmr;
        }

        public final List<Integer> getPredictions() {
            return this.predictions;
        }

        public final void addPrediction(int i) {
            this.predictions.add(Integer.valueOf(i));
        }

        public final void deletePredictions() {
            this.predictions.clear();
        }
    }

    static /* synthetic */ void groupAndAddBoxesByField$default(Vision vision, FormConfig.Field field, List list, List list2, String str, List list3, int i, Object obj) {
        if ((i & 16) != 0) {
            list3 = null;
        }
        vision.groupAndAddBoxesByField(field, list, list2, str, list3);
    }

    private final void groupAndAddBoxesByField(FormConfig.Field field, List<BoxGroup> list, List<? extends Rect> list2, String str, List<? extends Rect> list3) {
        BoxGroup boxGroup;
        String name = field.getName();
        int size = field.getCoordinates().size();
        if (list2.size() >= size) {
            boolean z = false;
            List<? extends Rect> subList = list2.subList(0, size);
            Collection collection = list3;
            if (collection == null || collection.isEmpty()) {
                z = true;
            }
            if (z) {
                boxGroup = new BoxGroup(name, subList, str, CollectionsKt.emptyList());
            } else {
                boxGroup = new BoxGroup(name, subList, str, list3);
            }
            list.add(boxGroup);
            return;
        }
        throw new IllegalArgumentException("Insufficient number of '" + str + "' boxes found for field: " + name);
    }

    private final Mat applyAdaptiveThreshold(Mat mat) {
        CLAHE createCLAHE = Imgproc.createCLAHE(adjustClipLimit(calculateImageContrast(mat)), new Size(RangesKt.coerceAtLeast(((double) mat.width()) * 0.02d, 1.0d), RangesKt.coerceAtLeast(((double) mat.width()) * 0.02d, 1.0d)));
        Mat mat2 = new Mat();
        createCLAHE.apply(mat, mat2);
        int max = Math.max((int) (((double) Math.min(mat2.rows(), mat2.cols())) * 0.01d), 3);
        if (max % 2 == 0) {
            max++;
        }
        int i = max;
        int max2 = Math.max((int) (((double) i) * 0.1d), 1);
        Mat mat3 = new Mat();
        Imgproc.adaptiveThreshold(mat2, mat3, 255.0d, 1, 1, i, (double) max2);
        return mat3;
    }

    private final double calculateImageContrast(Mat mat) {
        MatOfDouble matOfDouble = new MatOfDouble();
        Core.meanStdDev(mat, new MatOfDouble(), matOfDouble);
        return matOfDouble.toArray()[0];
    }

    public void close() {
        List<Interpreter> list = this.interpreters;
        Interpreter interpreter = null;
        if (list == null) {
            Intrinsics.throwUninitializedPropertyAccessException("interpreters");
            list = null;
        }
        for (Interpreter close : list) {
            close.close();
        }
        Interpreter interpreter2 = this.interpretersBlank;
        if (interpreter2 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("interpretersBlank");
        } else {
            interpreter = interpreter2;
        }
        interpreter.close();
    }

    @Metadata(d1 = {"\u0000\"\n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0002\b\u0002\n\u0002\u0010\u0006\n\u0002\b\u0002\n\u0002\u0018\u0002\n\u0002\b\u0006\n\u0002\u0010\u000e\n\u0000\b\u0003\u0018\u00002\u00020\u0001B\u0007\b\u0002¢\u0006\u0002\u0010\u0002R\u000e\u0010\u0003\u001a\u00020\u0004XT¢\u0006\u0002\n\u0000R\u000e\u0010\u0005\u001a\u00020\u0004XT¢\u0006\u0002\n\u0000R\u000e\u0010\u0006\u001a\u00020\u0007X\u0004¢\u0006\u0002\n\u0000R\u000e\u0010\b\u001a\u00020\u0007X\u0004¢\u0006\u0002\n\u0000R\u000e\u0010\t\u001a\u00020\u0007X\u0004¢\u0006\u0002\n\u0000R\u000e\u0010\n\u001a\u00020\u0004XT¢\u0006\u0002\n\u0000R\u000e\u0010\u000b\u001a\u00020\u0004XT¢\u0006\u0002\n\u0000R\u000e\u0010\f\u001a\u00020\u0004XT¢\u0006\u0002\n\u0000R\u000e\u0010\r\u001a\u00020\u000eXT¢\u0006\u0002\n\u0000¨\u0006\u000f"}, d2 = {"Lorg/informatika/sirekap/support/vision/Vision$Companion;", "", "()V", "BOX_SIZE_TOLERANCE_OFFSET", "", "CIRCLE_TO_CROPPED_WIDTH_RATIO", "COLOR_BLUE", "Lorg/opencv/core/Scalar;", "COLOR_GREEN", "COLOR_RED", "OMR_BOX_SIZE_TOLERANCE_OFFSET", "ROI_SIZE_NEGATIVE_OFFSET", "ROI_SIZE_POSITIVE_OFFSET", "TAG", "", "app_productionRelease"}, k = 1, mv = {1, 8, 0}, xi = 48)
    /* compiled from: Vision.kt */
    public static final class Companion {
        public /* synthetic */ Companion(DefaultConstructorMarker defaultConstructorMarker) {
            this();
        }

        private Companion() {
        }
    }
}
