package com.sdxlnpu.app;

import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import android.content.Intent;
import android.content.SharedPreferences;
import android.view.Menu;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SDXL Lightning on Qualcomm NPU — standalone phone generation.
 *
 * Calls phone_generate.py (Termux Python) via su, parses stdout for progress,
 * loads the resulting PNG and displays it. No JNI, no PC required.
 */
public class MainActivity extends AppCompatActivity {

    private String BASE_DIR;
    private String GEN_SCRIPT;
    private String OUTPUT_DIR;
    private String PYTHON;

    private EditText promptInput;
    private EditText negPromptInput;
    private EditText seedInput;
    private SeekBar stepsSeekBar;
    private TextView stepsLabel;
    private SeekBar cfgSeekBar;
    private TextView cfgLabel;
    private CheckBox contrastStretch;
    private MaterialButton generateButton;
    private MaterialButton saveButton;
    private MaterialButton stopButton;
    private ProgressBar progressBar;
    private TextView statusText;
    private TextView timingText;
    private ImageView imagePreview;

    private ExecutorService executor;
    private Handler mainHandler;
    private Bitmap currentBitmap;
    private volatile Process currentProcess;

    // Patterns for parsing generate.py stdout
    private static final Pattern PAT_CLIP = Pattern.compile("\\[CLIP (cond|uncond)\\].*?L=(\\d+)ms G=(\\d+)ms");
    private static final Pattern PAT_UNET = Pattern.compile("\\[UNet (\\d+)/(\\d+)\\]\\s+(\\d+)ms");
    private static final Pattern PAT_VAE = Pattern.compile("\\[VAE\\]\\s+(\\d+)ms");
    private static final Pattern PAT_SAVED = Pattern.compile("Saved:\\s+(.+\\.png)");
    private static final Pattern PAT_TOTAL = Pattern.compile("Total:\\s+([\\d.]+)s");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        promptInput = findViewById(R.id.promptInput);
        negPromptInput = findViewById(R.id.negPromptInput);
        seedInput = findViewById(R.id.seedInput);
        stepsSeekBar = findViewById(R.id.stepsSeekBar);
        stepsLabel = findViewById(R.id.stepsLabel);
        cfgSeekBar = findViewById(R.id.cfgSeekBar);
        cfgLabel = findViewById(R.id.cfgLabel);
        contrastStretch = findViewById(R.id.contrastStretch);
        generateButton = findViewById(R.id.generateButton);
        saveButton = findViewById(R.id.saveButton);
        stopButton = findViewById(R.id.stopButton);
        progressBar = findViewById(R.id.progressBar);
        statusText = findViewById(R.id.statusText);
        timingText = findViewById(R.id.timingText);
        imagePreview = findViewById(R.id.imagePreview);

        executor = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());

        loadSettings();

        stepsSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                stepsLabel.setText(String.format(Locale.US, "Steps: %d", progress));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        cfgSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                cfgLabel.setText(String.format(Locale.US, "CFG: %.1f", progress / 10f));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        generateButton.setOnClickListener(v -> startGeneration());
        saveButton.setOnClickListener(v -> saveImage());
        stopButton.setOnClickListener(v -> stopGeneration());

        // Check prerequisites
        checkPrerequisites();
    }

    private void loadSettings() {
        SharedPreferences prefs = getSharedPreferences(
            SettingsActivity.PREFS_NAME, MODE_PRIVATE);
        BASE_DIR = prefs.getString(SettingsActivity.KEY_BASE_DIR,
            SettingsActivity.DEFAULT_BASE_DIR);
        PYTHON = prefs.getString(SettingsActivity.KEY_PYTHON_PATH,
            SettingsActivity.DEFAULT_PYTHON);
        GEN_SCRIPT = BASE_DIR + "/phone_gen/generate.py";
        OUTPUT_DIR = BASE_DIR + "/outputs";
    }

    private void checkPrerequisites() {
        File ctx = new File(BASE_DIR, "context");
        if (!ctx.exists()) {
            statusText.setText("Модели не найдены в " + BASE_DIR +
                "\nДеплойте через scripts/deploy_to_phone.py" +
                "\nили измените путь в Настройках (⚙)");
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, 1, 0, R.string.settings)
            .setIcon(android.R.drawable.ic_menu_preferences)
            .setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == 1) {
            Intent intent = new Intent(this, SettingsActivity.class);
            startActivityForResult(intent, 100);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 100 && resultCode == RESULT_OK) {
            loadSettings();
            checkPrerequisites();
        }
    }

    private void startGeneration() {
        String prompt = promptInput.getText().toString().trim();
        if (prompt.isEmpty()) {
            Toast.makeText(this, "Введите промпт", Toast.LENGTH_SHORT).show();
            return;
        }

        String seedStr = seedInput.getText().toString().trim();
        long seed = seedStr.isEmpty()
            ? (new Random().nextInt(900000) + 100000)
            : Long.parseLong(seedStr);
        int steps = stepsSeekBar.getProgress();
        float cfg = cfgSeekBar.getProgress() / 10f;
        String neg = negPromptInput.getText().toString().trim();
        boolean stretch = contrastStretch.isChecked();

        // Build output name
        String outName = "apk_s" + seed;

        generateButton.setEnabled(false);
        stopButton.setVisibility(View.VISIBLE);
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setProgress(0);
        saveButton.setVisibility(View.GONE);
        timingText.setVisibility(View.GONE);
        statusText.setText("Запуск...");

        executor.execute(() -> {
            try {
                runPipeline(prompt, seed, steps, cfg, neg, stretch, outName);
            } catch (Exception e) {
                mainHandler.post(() -> {
                    statusText.setText("Ошибка: " + e.getMessage());
                    generateButton.setEnabled(true);
                    stopButton.setVisibility(View.GONE);
                    progressBar.setVisibility(View.GONE);
                });
            }
        });
    }

    private void stopGeneration() {
        Process p = currentProcess;
        if (p != null) {
            p.destroyForcibly();
            currentProcess = null;
        }
        mainHandler.post(() -> {
            statusText.setText("Остановлено");
            generateButton.setEnabled(true);
            stopButton.setVisibility(View.GONE);
            progressBar.setVisibility(View.GONE);
        });
    }

    private void updateStatus(String status, int progress) {
        mainHandler.post(() -> {
            statusText.setText(status);
            if (progress >= 0) progressBar.setProgress(progress);
        });
    }

    private void runPipeline(String prompt, long seed, int steps,
                             float cfg, String neg, boolean stretch,
                             String outName) throws IOException, InterruptedException {
        // Build shell script (multi-line — no nested-quote issues)
        StringBuilder script = new StringBuilder();
        script.append("export PATH=/data/data/com.termux/files/usr/bin:$PATH\n");
        script.append(String.format(Locale.US,
            "export LD_LIBRARY_PATH=%s/lib:%s/bin:$LD_LIBRARY_PATH\n", BASE_DIR, BASE_DIR));
        script.append(String.format(Locale.US,
            "export ADSP_LIBRARY_PATH=\"%s/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp\"\n", BASE_DIR));
        script.append("cd ").append(BASE_DIR).append("\n");

        script.append(PYTHON).append(" ").append(GEN_SCRIPT);
        script.append(" \"").append(shellEscape(prompt)).append("\"");
        script.append(" --seed ").append(seed);
        script.append(" --steps ").append(steps);
        script.append(" --name ").append(outName);
        if (cfg > 1.0f) {
            script.append(" --cfg ").append(String.format(Locale.US, "%.1f", cfg));
            if (!neg.isEmpty()) {
                script.append(" --neg \"").append(shellEscape(neg)).append("\"");
            }
        }
        if (!stretch) {
            script.append(" --no-stretch");
        }
        script.append(" 2>&1\n");

        updateStatus("Запуск...", 2);

        // Find su and launch root shell in global mount namespace
        // --mount-master is needed because app runs in isolated mount namespace
        // where /data/data/com.termux/ is not visible
        String suBin = findSu();
        ProcessBuilder pb = new ProcessBuilder(suBin, "--mount-master");
        pb.redirectErrorStream(true);
        Process process = pb.start();
        currentProcess = process;

        // Write script to su stdin — avoids all quoting problems
        try (OutputStream os = process.getOutputStream()) {
            os.write(script.toString().getBytes("UTF-8"));
            os.flush();
        }

        StringBuilder timingLog = new StringBuilder();
        StringBuilder rawLog = new StringBuilder();
        String savedPath = null;
        int clipDone = 0;

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (currentProcess == null) break; // stopped
                if (rawLog.length() < 4000) rawLog.append(line).append("\n");

                // Parse CLIP progress
                Matcher m = PAT_CLIP.matcher(line);
                if (m.find()) {
                    clipDone++;
                    String kind = m.group(1);
                    int msL = Integer.parseInt(m.group(2));
                    int msG = Integer.parseInt(m.group(3));
                    timingLog.append(String.format(Locale.US,
                        "CLIP %s: L=%dms G=%dms\n", kind, msL, msG));
                    updateStatus("CLIP " + kind + "... " + (msL + msG) + "ms",
                        5 + clipDone * 3);
                    continue;
                }

                // Parse UNet steps
                m = PAT_UNET.matcher(line);
                if (m.find()) {
                    int step = Integer.parseInt(m.group(1));
                    int total = Integer.parseInt(m.group(2));
                    int ms = Integer.parseInt(m.group(3));
                    timingLog.append(String.format(Locale.US,
                        "  UNet %d/%d: %dms\n", step, total, ms));
                    int pct = 10 + step * 75 / total;
                    updateStatus(String.format(Locale.US,
                        "UNet %d/%d — %dms", step, total, ms), pct);
                    continue;
                }

                // Parse VAE
                m = PAT_VAE.matcher(line);
                if (m.find()) {
                    int ms = Integer.parseInt(m.group(1));
                    timingLog.append(String.format(Locale.US, "VAE: %dms\n", ms));
                    updateStatus("VAE... " + ms + "ms", 90);
                    continue;
                }

                // Parse saved path
                m = PAT_SAVED.matcher(line);
                if (m.find()) {
                    savedPath = m.group(1).trim();
                    continue;
                }

                // Parse total time
                m = PAT_TOTAL.matcher(line);
                if (m.find()) {
                    timingLog.append("Total: ").append(m.group(1)).append("s\n");
                    continue;
                }

                // UNet total line
                if (line.contains("UNet total:")) {
                    timingLog.append(line.trim()).append("\n");
                }
            }
        }

        int exitCode = process.waitFor();
        currentProcess = null;

        if (exitCode != 0 && savedPath == null) {
            String hint;
            if (exitCode == 127) {
                hint = "Команда не найдена (код 127).\nПроверьте: установлен ли Termux с Python3?";
            } else if (exitCode == 1 || exitCode == 13) {
                hint = "Root-доступ не предоставлен.\nОткройте Magisk и разрешите root для SDXL NPU.";
            } else {
                hint = "Генерация завершилась с ошибкой (код " + exitCode + ")";
            }
            String details = rawLog.length() > 0
                ? "\n\n" + rawLog.toString().trim() : "";
            throw new IOException(hint + details);
        }

        // Load the generated PNG
        final String finalPath = savedPath != null
            ? savedPath
            : OUTPUT_DIR + "/" + outName + ".png";

        File pngFile = new File(finalPath);
        if (!pngFile.exists()) {
            throw new IOException("Файл не найден: " + finalPath);
        }

        Bitmap bitmap = BitmapFactory.decodeFile(finalPath);
        if (bitmap == null) {
            throw new IOException("Не удалось загрузить изображение: " + finalPath);
        }

        final String finalTiming = timingLog.toString();
        mainHandler.post(() -> {
            currentBitmap = bitmap;
            imagePreview.setImageBitmap(bitmap);
            saveButton.setVisibility(View.VISIBLE);
            stopButton.setVisibility(View.GONE);
            progressBar.setVisibility(View.GONE);
            generateButton.setEnabled(true);
            statusText.setText("Готово!");
            timingText.setText(finalTiming);
            timingText.setVisibility(View.VISIBLE);
        });
    }

    /** Escape string for use inside double-quoted shell argument */
    private static String shellEscape(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("$", "\\$")
                .replace("`", "\\`");
    }

    /** Find su binary (Magisk / KernelSU / other root solutions) */
    private static String findSu() {
        for (String path : new String[]{
            "/product/bin/su",
            "/sbin/su", "/system/xbin/su", "/system/bin/su",
            "/su/bin/su", "/data/adb/magisk/su"
        }) {
            if (new java.io.File(path).exists()) return path;
        }
        return "su"; // fallback: try PATH
    }

    private void saveImage() {
        if (currentBitmap == null) return;

        ContentValues values = new ContentValues();
        String filename = "sdxl_npu_" + System.currentTimeMillis() + ".png";
        values.put(MediaStore.Images.Media.DISPLAY_NAME, filename);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.RELATIVE_PATH,
            Environment.DIRECTORY_PICTURES + "/SDXL_NPU");

        Uri uri = getContentResolver().insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        if (uri != null) {
            try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                currentBitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                Toast.makeText(this, "Сохранено: " + filename, Toast.LENGTH_LONG).show();
            } catch (IOException e) {
                Toast.makeText(this, "Ошибка сохранения: " + e.getMessage(),
                    Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Process p = currentProcess;
        if (p != null) p.destroyForcibly();
        executor.shutdown();
    }
}
