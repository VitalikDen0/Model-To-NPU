package com.sdxlnpu.app;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.view.MenuItem;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.textfield.TextInputEditText;

import java.io.File;

/**
 * Settings for model paths and phone-side configuration.
 * Allows user to change the base directory where context binaries
 * and phone_generate.py are located on the phone filesystem.
 */
public class SettingsActivity extends AppCompatActivity {

    public static final String PREFS_NAME = "sdxl_npu_prefs";
    public static final String KEY_BASE_DIR = "base_dir";
    public static final String KEY_ROOT_BASE_DIR = "root_base_dir";
    public static final String KEY_PYTHON_PATH = "python_path";
    public static final String DOWNLOADS_BASE_DIR = "/sdcard/Download/sdxl_qnn";
    public static final String LEGACY_BASE_DIR = "/data/local/tmp/sdxl_qnn";
    public static final String DEFAULT_ROOT_BASE_DIR = LEGACY_BASE_DIR;
    public static final String WAN_DOWNLOADS_BASE_DIR = "/sdcard/Download/wan21_t2v_qnn";
    public static final String WAN_LEGACY_BASE_DIR = "/data/local/tmp/wan21_t2v_qnn";
    public static final String DEFAULT_BASE_DIR = DOWNLOADS_BASE_DIR;
    public static final String DEFAULT_PYTHON = "python3";
    public static final String LEGACY_PYTHON = "/data/data/com.termux/files/usr/bin/python3";

    private TextInputEditText baseDirInput;
    private TextInputEditText rootBaseDirInput;
    private TextInputEditText pythonPathInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        androidx.appcompat.widget.Toolbar toolbar = findViewById(R.id.settingsToolbar);
        if (toolbar != null) {
            toolbar.setNavigationOnClickListener(v -> finish());
        } else if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle(R.string.settings_title);
        }

        baseDirInput = findViewById(R.id.baseDirInput);
        rootBaseDirInput = findViewById(R.id.rootBaseDirInput);
        pythonPathInput = findViewById(R.id.pythonPathInput);

        MaterialButton saveBtn = findViewById(R.id.saveSettingsButton);
        MaterialButton resetBtn = findViewById(R.id.resetSettingsButton);
        MaterialButton verifyBtn = findViewById(R.id.verifyButton);

        // Load saved preferences
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String detectedBaseDir = detectDefaultBaseDir();
        String detectedPython = detectDefaultPython(detectedBaseDir);
        baseDirInput.setText(prefs.getString(KEY_BASE_DIR, detectedBaseDir));
        rootBaseDirInput.setText(prefs.getString(KEY_ROOT_BASE_DIR, DEFAULT_ROOT_BASE_DIR));
        pythonPathInput.setText(prefs.getString(KEY_PYTHON_PATH, detectedPython));

        saveBtn.setOnClickListener(v -> saveSettings());
        resetBtn.setOnClickListener(v -> resetSettings());
        verifyBtn.setOnClickListener(v -> verifySetup());
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void saveSettings() {
        String baseDir = baseDirInput.getText().toString().trim();
        String rootBaseDir = rootBaseDirInput.getText().toString().trim();
        String python = pythonPathInput.getText().toString().trim();

        if (baseDir.isEmpty()) baseDir = detectDefaultBaseDir();
        if (python.isEmpty()) python = detectDefaultPython(baseDir);

        // Basic validation — no command injection
        if (baseDir.contains(";") || baseDir.contains("&") || baseDir.contains("|")
            || rootBaseDir.contains(";") || rootBaseDir.contains("&") || rootBaseDir.contains("|")
            || python.contains(";") || python.contains("&") || python.contains("|")) {
            Toast.makeText(this, "Недопустимые символы в пути", Toast.LENGTH_SHORT).show();
            return;
        }

        SharedPreferences.Editor editor = getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit();
        editor.putString(KEY_BASE_DIR, baseDir);
        editor.putString(KEY_ROOT_BASE_DIR, rootBaseDir.isEmpty() ? DEFAULT_ROOT_BASE_DIR : rootBaseDir);
        editor.putString(KEY_PYTHON_PATH, python);
        editor.apply();

        Toast.makeText(this, "Настройки сохранены", Toast.LENGTH_SHORT).show();
        setResult(RESULT_OK);
        finish();
    }

    private void resetSettings() {
        String detectedBaseDir = detectDefaultBaseDir();
        baseDirInput.setText(detectedBaseDir);
        rootBaseDirInput.setText(DEFAULT_ROOT_BASE_DIR);
        pythonPathInput.setText(detectDefaultPython(detectedBaseDir));
        Toast.makeText(this, "Сброшено к значениям по умолчанию", Toast.LENGTH_SHORT).show();
    }

    private void verifySetup() {
        String baseDir = baseDirInput.getText().toString().trim();
        if (baseDir.isEmpty()) baseDir = detectDefaultBaseDir();

        if (baseDir.contains(";") || baseDir.contains("&") || baseDir.contains("|")) {
            Toast.makeText(this, "Недопустимые символы в пути", Toast.LENGTH_SHORT).show();
            return;
        }

        File base = new File(baseDir);
        File contextDir = new File(base, "context");

        // Inflate custom modern dialog layout
        android.view.LayoutInflater inflater = android.view.LayoutInflater.from(this);
        android.view.View dialogView = inflater.inflate(R.layout.dialog_file_verification, null);
        android.widget.TextView baseDirText = dialogView.findViewById(R.id.dialogBaseDirText);
        android.widget.TextView overallBadge = dialogView.findViewById(R.id.dialogOverallBadge);
        android.widget.LinearLayout itemsContainer = dialogView.findViewById(R.id.dialogItemsContainer);
        com.google.android.material.button.MaterialButton closeBtn = dialogView.findViewById(R.id.dialogCloseButton);

        baseDirText.setText(baseDir);

        boolean allOk = true;

        // 1. Storage Permission
        boolean storageAccess = Build.VERSION.SDK_INT < Build.VERSION_CODES.R || Environment.isExternalStorageManager();
        addVerificationRow(itemsContainer, inflater, "Доступ к файловой системе",
            storageAccess ? "Полный доступ (MANAGE_EXTERNAL_STORAGE)" : "Отсутствует разрешение на чтение/запись",
            storageAccess, "ДОСТУП ЕСТЬ", "НЕТ");
        if (!storageAccess) allOk = false;

        // 2. UNet Model
        File unetBase = new File(contextDir, "unet_base.serialized.bin.bin");
        File unet1024 = new File(contextDir, "unet_base_1024x1024.serialized.bin.bin");
        File unetEnc = new File(contextDir, "unet_encoder_fp16.serialized.bin.bin");
        File unetDec = new File(contextDir, "unet_decoder_fp16.serialized.bin.bin");
        boolean hasBaseUnet = unetBase.isFile() || unet1024.isFile();
        boolean hasSplitUnet = unetEnc.isFile() && unetDec.isFile();
        if (hasBaseUnet) {
            File unetFile = unetBase.isFile() ? unetBase : unet1024;
            double sizeGb = unetFile.length() / (1024.0 * 1024.0 * 1024.0);
            addVerificationRow(itemsContainer, inflater, "Base UNet (W8A16 NPU)",
                String.format(java.util.Locale.US, "%s (%.2f ГБ)", unetFile.getName(), sizeGb),
                true, "НАЙДЕНО", "НЕТ");
        } else if (hasSplitUnet) {
            addVerificationRow(itemsContainer, inflater, "Lightning UNet (Split NPU)",
                "unet_encoder_fp16 + unet_decoder_fp16",
                true, "НАЙДЕНО", "НЕТ");
        } else {
            addVerificationRow(itemsContainer, inflater, "UNet Граф нейросети",
                "Не найден unet_base.serialized.bin.bin в context/",
                false, "НАЙДЕНО", "ОТСУТСТВУЕТ");
            allOk = false;
        }

        // 3. VAE Decoder
        File vaeFile = new File(contextDir, "vae_decoder.serialized.bin.bin");
        addVerificationRow(itemsContainer, inflater, "VAE Декодер",
            vaeFile.isFile() ? String.format(java.util.Locale.US, "vae_decoder (%.1f МБ)", vaeFile.length() / (1024.0 * 1024.0)) : "context/vae_decoder.serialized.bin.bin",
            vaeFile.isFile(), "НАЙДЕНО", "ОТСУТСТВУЕТ");
        if (!vaeFile.isFile()) allOk = false;

        // 4. CLIP Encoders
        File clipL = new File(contextDir, "clip_l.serialized.bin.bin");
        File clipG = new File(contextDir, "clip_g.serialized.bin.bin");
        boolean hasClip = clipL.isFile() && clipG.isFile();
        addVerificationRow(itemsContainer, inflater, "CLIP Энкодеры (Text)",
            hasClip ? "clip_l + clip_g (NPU)" : "Требуются clip_l и clip_g в context/",
            hasClip, "НАЙДЕНО", "ОТСУТСТВУЕТ");
        if (!hasClip) allOk = false;

        // 5. Native C-Engine Binary
        File bundledPayload = RuntimeBootstrap.getBundledRuntimePayloadDir(this);
        File bundledServer = bundledPayload != null ? new File(bundledPayload, "bin/qnn-multi-context-server") : null;
        File stagedServer = new File("/data/local/tmp/sdxl_app_engine/bin/qnn-multi-context-server");
        File extServer = new File(base, "bin/qnn-multi-context-server");
        boolean hasEngine = (bundledServer != null && bundledServer.isFile()) || stagedServer.isFile() || extServer.isFile() || new File("/data/local/tmp/qnn-multi-context-server").isFile();
        addVerificationRow(itemsContainer, inflater, "Native C-Engine (Zero-Root)",
            hasEngine ? "qnn-multi-context-server (NDK Clang 35, V79 HTP)" : "Движок не найден в payload / tmp",
            hasEngine, "ГОТОВО", "НЕТ");
        if (!hasEngine) allOk = false;

        // 6. QNN HTP Runtime Libs
        File bundledHtp = bundledPayload != null ? new File(bundledPayload, "lib/libQnnHtp.so") : null;
        File stagedHtp = new File("/data/local/tmp/sdxl_app_engine/lib/libQnnHtp.so");
        File extHtp = new File(base, "lib/libQnnHtp.so");
        boolean hasHtp = (bundledHtp != null && bundledHtp.isFile()) || stagedHtp.isFile() || extHtp.isFile();
        addVerificationRow(itemsContainer, inflater, "Qualcomm QNN HTP Backend",
            hasHtp ? "libQnnHtp.so + libcdsprpc.so (Hexagon V79 HTP)" : "libQnnHtp.so не найдена",
            hasHtp, "ГОТОВО", "НЕТ");
        if (!hasHtp) allOk = false;

        // 7. Tokenizer
        File bundledTok = bundledPayload != null ? new File(bundledPayload, "phone_gen/tokenizer/vocab.json") : null;
        File extTok = new File(base, "tokenizer/vocab.json");
        boolean hasTok = (bundledTok != null && bundledTok.isFile()) || extTok.isFile();
        addVerificationRow(itemsContainer, inflater, "SDXL Токенизатор",
            hasTok ? "vocab.json + merges.txt" : "vocab.json не найден",
            hasTok, "ГОТОВО", "НЕТ");
        if (!hasTok) allOk = false;

        // 8. LoRA Verification: Separate ready .bin.bin from raw .safetensors
        File loraDir1 = new File(base, "lora");
        File loraDir2 = new File(base, "loras");
        File loraSlots = new File(contextDir, "lora_slots");
        java.util.List<String> compiledLoras = new java.util.ArrayList<>();
        java.util.List<String> rawSafetensors = new java.util.ArrayList<>();

        for (File ld : new File[]{loraDir1, loraDir2}) {
            if (ld.isDirectory()) {
                File[] files = ld.listFiles();
                if (files != null) {
                    for (File f : files) {
                        String fn = f.getName();
                        if (f.isFile()) {
                            if (fn.endsWith(".bin.bin") || fn.endsWith(".serialized.bin")) {
                                compiledLoras.add(fn);
                            } else if (fn.endsWith(".safetensors")) {
                                rawSafetensors.add(fn);
                            }
                        }
                    }
                }
            }
        }
        if (loraSlots.isDirectory()) {
            File[] slotDirs = loraSlots.listFiles(File::isDirectory);
            if (slotDirs != null) {
                for (File sd : slotDirs) {
                    File[] sf = sd.listFiles((d, n) -> n.endsWith(".bin.bin") || n.endsWith(".serialized.bin"));
                    if (sf != null && sf.length > 0) {
                        compiledLoras.add(sd.getName() + " (слот)");
                    }
                }
            }
        }

        // Ready NPU LoRA Row (.bin.bin)
        if (!compiledLoras.isEmpty()) {
            addVerificationRow(itemsContainer, inflater, "LoRA Адаптеры (NPU .bin.bin)",
                "Готовы к генерации: " + String.join(", ", compiledLoras),
                0xFF10B981, String.format(java.util.Locale.US, "%d ГОТОВО", compiledLoras.size()));
        } else {
            addVerificationRow(itemsContainer, inflater, "LoRA Адаптеры (NPU .bin.bin)",
                "Скомпилированные .bin.bin не найдены (в меню доступен 'None')",
                0xFF94A3B8, "0");
        }

        // Raw .safetensors Row
        if (!rawSafetensors.isEmpty()) {
            addVerificationRow(itemsContainer, inflater, "Исходники LoRA (.safetensors)",
                "Найдено: " + String.join(", ", rawSafetensors) + "\n⚠️ Не являются .bin.bin (требуют компиляции на ПК)",
                0xFFF59E0B, String.format(java.util.Locale.US, "%d СЫРЫЕ", rawSafetensors.size()));
        } else {
            boolean loraDirExists = loraDir1.isDirectory() || loraDir2.isDirectory();
            addVerificationRow(itemsContainer, inflater, "Исходники LoRA (.safetensors)",
                loraDirExists ? "Файлы .safetensors не найдены" : "Папка /lora/ отсутствует",
                0xFF94A3B8, "0");
        }

        if (allOk) {
            overallBadge.setText("ГОТОВО К ГЕНЕРАЦИИ");
            overallBadge.setTextColor(0xFF10B981);
        } else {
            overallBadge.setText("ТРЕБУЮТСЯ ФАЙЛЫ");
            overallBadge.setTextColor(0xFFF43F5E);
        }

        androidx.appcompat.app.AlertDialog dialog = new androidx.appcompat.app.AlertDialog.Builder(this)
            .setView(dialogView)
            .create();

        closeBtn.setOnClickListener(v -> dialog.dismiss());
        dialog.show();
    }

    private void addVerificationRow(android.widget.LinearLayout container, android.view.LayoutInflater inflater,
                                    String title, String detail, boolean ok, String okText, String failText) {
        addVerificationRow(container, inflater, title, detail, ok ? 0xFF10B981 : 0xFFF43F5E, ok ? okText : failText);
    }

    private void addVerificationRow(android.widget.LinearLayout container, android.view.LayoutInflater inflater,
                                    String title, String detail, int chipColor, String chipText) {
        android.view.View row = inflater.inflate(R.layout.item_verification_row, container, false);
        android.widget.TextView titleTv = row.findViewById(R.id.itemTitle);
        android.widget.TextView detailTv = row.findViewById(R.id.itemDetail);
        android.widget.TextView statusChip = row.findViewById(R.id.itemStatusChip);

        titleTv.setText(title);
        detailTv.setText(detail);
        statusChip.setText(chipText != null ? chipText : "OK");
        statusChip.setTextColor(chipColor);
        container.addView(row);
    }

    public static String detectDefaultBaseDir() {
        if (new File(LEGACY_BASE_DIR).exists()) {
            return LEGACY_BASE_DIR;
        }
        if (new File(DOWNLOADS_BASE_DIR).exists()) {
            return DOWNLOADS_BASE_DIR;
        }
        return DEFAULT_BASE_DIR;
    }

    public static String detectDefaultPython(String baseDir) {
        return baseDir != null && (baseDir.startsWith(LEGACY_BASE_DIR) || baseDir.startsWith(WAN_LEGACY_BASE_DIR))
            ? LEGACY_PYTHON
            : DEFAULT_PYTHON;
    }
}
