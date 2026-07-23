package com.sdxlnpu.app;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

final class RuntimeBootstrap {

    private static final String TAG = "SDXLNPU";
    private static final String ASSET_ROOT = "termux_bundle";
    private static final String RUNTIME_PAYLOAD_DIR = "runtime_payload";
    private static final String VERSION_MARKER = ".bundle_version";
    private static final String RUNTIME_PAYLOAD_VERSION_MARKER = "runtime_payload_version.txt";
    private static final String BUNDLE_LAYOUT_VERSION = "termux-bundle-v3";
    private static final int COPY_BUFFER_SIZE = 1024 * 1024;

    private RuntimeBootstrap() {
    }

    static boolean hasBundledAssets(Context context) {
        try {
            String[] children = context.getAssets().list(ASSET_ROOT);
            return children != null && children.length > 0;
        } catch (IOException e) {
            return false;
        }
    }

    static File getBundleDir(Context context) {
        return new File(context.getFilesDir(), "termux_bundle");
    }

    static File getBundledPrefixDir(Context context) {
        return new File(getBundleDir(context), "prefix");
    }

    static File getBundledRuntimePayloadDir(Context context) {
        return new File(getBundleDir(context), RUNTIME_PAYLOAD_DIR);
    }

    static String findBundledPython(Context context) {
        File prefix = getBundledPrefixDir(context);
        File[] candidates = new File[] {
            new File(prefix, "bin/python3"),
            new File(prefix, "bin/python"),
        };
        for (File candidate : candidates) {
            if (candidate.isFile() && candidate.canExecute()) {
                return candidate.getAbsolutePath();
            }
        }
        return null;
    }

    static String ensureBundledAssetsExtracted(Context context) throws IOException {
        if (!hasBundledAssets(context)) {
            Log.w(TAG, "Bundled assets not present in APK");
            return null;
        }

        File bundleDir = getBundleDir(context);
        String expectedVersion = BUNDLE_LAYOUT_VERSION;
        String expectedRuntimePayloadVersion = readAssetTextFileOrNull(
            context.getAssets(),
            ASSET_ROOT + "/" + RUNTIME_PAYLOAD_DIR + "/" + RUNTIME_PAYLOAD_VERSION_MARKER
        );
        File marker = new File(bundleDir, VERSION_MARKER);
        if (bundleDir.isDirectory() && marker.isFile()) {
            String currentVersion = readTextFile(marker).trim();
            boolean layoutMatches = expectedVersion.equals(currentVersion);
            boolean payloadMatches = true;
            if (expectedRuntimePayloadVersion != null && !expectedRuntimePayloadVersion.isEmpty()) {
                File runtimePayloadMarker = new File(
                    new File(bundleDir, RUNTIME_PAYLOAD_DIR),
                    RUNTIME_PAYLOAD_VERSION_MARKER
                );
                payloadMatches = runtimePayloadMarker.isFile()
                    && expectedRuntimePayloadVersion.equals(readTextFile(runtimePayloadMarker).trim());
            }
            if (layoutMatches && payloadMatches) {
                Log.i(TAG, "Bundled assets already extracted: layout=" + currentVersion
                    + ", payload=" + expectedRuntimePayloadVersion);
                return bundleDir.getAbsolutePath();
            }
            Log.i(TAG, "Bundled assets need refresh: currentLayout=" + currentVersion
                + ", expectedLayout=" + expectedVersion
                + ", expectedPayload=" + expectedRuntimePayloadVersion);
        }

        Log.i(TAG, "Extracting bundled assets into " + bundleDir.getAbsolutePath());

        try {
            deleteRecursively(bundleDir);
        } catch (IOException e) {
            Log.w(TAG, "Normal bundled runtime cleanup failed, trying root fallback", e);
            if (!tryDeleteRecursivelyAsRoot(bundleDir)) {
                throw e;
            }
        }
        if (!bundleDir.mkdirs() && !bundleDir.isDirectory()) {
            throw new IOException("Не удалось создать каталог bundled runtime: " + bundleDir);
        }

        copyAssetTree(context.getAssets(), ASSET_ROOT, bundleDir);
        writeTextFile(marker, expectedVersion);

        // Set executable permissions on prefix/bin/* and prefix/lib/*.so
        setExecutablePermissions(bundleDir);

        Log.i(TAG, "Bundled assets extracted successfully");

        return bundleDir.getAbsolutePath();
    }

    private static void setExecutablePermissions(File bundleDir) {
        File binDir = new File(bundleDir, "prefix/bin");
        if (binDir.isDirectory()) {
            File[] files = binDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile()) f.setExecutable(true, false);
                }
            }
        }
        File libDir = new File(bundleDir, "prefix/lib");
        if (libDir.isDirectory()) {
            File[] files = libDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile() && f.getName().endsWith(".so"))
                        f.setExecutable(true, false);
                }
            }
        }

        File runtimeBinDir = new File(bundleDir, RUNTIME_PAYLOAD_DIR + "/bin");
        if (runtimeBinDir.isDirectory()) {
            File[] files = runtimeBinDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile()) {
                        f.setExecutable(true, false);
                    }
                }
            }
        }

        File runtimeLibDir = new File(bundleDir, RUNTIME_PAYLOAD_DIR + "/phone_gen/lib");
        if (runtimeLibDir.isDirectory()) {
            File[] files = runtimeLibDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile() && f.getName().endsWith(".so")) {
                        f.setExecutable(true, false);
                    }
                }
            }
        }

        File runtimeQnnLibDir = new File(bundleDir, RUNTIME_PAYLOAD_DIR + "/lib");
        if (runtimeQnnLibDir.isDirectory()) {
            File[] files = runtimeQnnLibDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile() && f.getName().endsWith(".so")) {
                        f.setExecutable(true, false);
                    }
                }
            }
        }
    }

    static String describeBundledAssets(Context context) {
        if (!hasBundledAssets(context)) {
            return "Bundled offline runtime: not packaged in this APK build";
        }
        try {
            String[] debs = context.getAssets().list(ASSET_ROOT + "/debs");
            String[] scripts = context.getAssets().list(ASSET_ROOT + "/scripts");
            String[] runtimePayload = context.getAssets().list(ASSET_ROOT + "/" + RUNTIME_PAYLOAD_DIR);
            int debCount = debs != null ? debs.length : 0;
            int scriptCount = scripts != null ? scripts.length : 0;
            int runtimePayloadCount = runtimePayload != null ? runtimePayload.length : 0;
            return "Bundled offline runtime: " + debCount + " debs, " + scriptCount
                + " scripts, runtime payload=" + runtimePayloadCount;
        } catch (IOException e) {
            return "Bundled offline runtime: available, but asset listing failed (" + e.getMessage() + ")";
        }
    }

    private static void copyAssetTree(AssetManager assetManager, String assetPath, File destination) throws IOException {
        String[] children = assetManager.list(assetPath);
        if (children == null || children.length == 0) {
            copySingleAsset(assetManager, assetPath, destination);
            return;
        }

        if (!destination.exists() && !destination.mkdirs() && !destination.isDirectory()) {
            throw new IOException("Не удалось создать каталог asset bundle: " + destination);
        }

        for (String child : children) {
            String childAssetPath = assetPath + "/" + child;
            File childDestination = new File(destination, child);
            copyAssetTree(assetManager, childAssetPath, childDestination);
        }
    }

    private static void copySingleAsset(AssetManager assetManager, String assetPath, File destination) throws IOException {
        File parent = destination.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs() && !parent.isDirectory()) {
            throw new IOException("Не удалось создать каталог для asset: " + parent);
        }

        try (InputStream in = assetManager.open(assetPath);
             FileOutputStream out = new FileOutputStream(destination)) {
            byte[] buffer = new byte[COPY_BUFFER_SIZE];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }

        if (destination.getName().endsWith(".sh")) {
            //noinspection ResultOfMethodCallIgnored
            destination.setExecutable(true, true);
        }
    }

    private static void deleteRecursively(File path) throws IOException {
        if (path == null || !path.exists()) {
            return;
        }
        if (path.isDirectory()) {
            File[] children = path.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!path.delete()) {
            throw new IOException("Не удалось удалить старый bundled runtime: " + path);
        }
    }

    private static boolean tryDeleteRecursivelyAsRoot(File path) {
        if (path == null || !path.exists()) {
            return true;
        }
        String su = findAvailableSuOrNull();
        if (su == null) {
            return false;
        }
        try {
            Process process = new ProcessBuilder(su, "--mount-master")
                .redirectErrorStream(true)
                .start();
            String script = "rm -rf '" + shellEscape(path.getAbsolutePath()) + "'\nexit\n";
            try (OutputStreamWriter writer = new OutputStreamWriter(
                    process.getOutputStream(), StandardCharsets.UTF_8)) {
                writer.write(script);
                writer.flush();
            }
            process.waitFor();
            return !path.exists();
        } catch (Exception e) {
            Log.w(TAG, "Root fallback cleanup failed", e);
            return false;
        }
    }

    private static String findAvailableSuOrNull() {
        for (String candidate : new String[] {
                "/product/bin/su",
                "/sbin/su", "/system/xbin/su", "/system/bin/su",
                "/su/bin/su", "/data/adb/magisk/su"
        }) {
            if (new File(candidate).exists()) {
                return candidate;
            }
        }
        return null;
    }

    private static String shellEscape(String value) {
        return value.replace("'", "'\\''");
    }

    private static String readTextFile(File file) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
        }
        return sb.toString();
    }

    private static String readAssetTextFileOrNull(AssetManager assetManager, String assetPath) {
        try (InputStream in = assetManager.open(assetPath);
             BufferedReader reader = new BufferedReader(
                 new InputStreamReader(in, StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
            return sb.toString().trim();
        } catch (IOException e) {
            return null;
        }
    }

    private static void writeTextFile(File file, String content) throws IOException {
        try (OutputStreamWriter writer = new OutputStreamWriter(
                new FileOutputStream(file), StandardCharsets.UTF_8)) {
            writer.write(content);
        }
    }

    // -------------------------------------------------------------------------
    // py_runtime: pre-built Python 3.13 + numpy + PIL bundled in py_runtime.zip
    // -------------------------------------------------------------------------

    private static final String PY_RUNTIME_ZIP_ASSET = "py_runtime.zip";
    static final String PY_RUNTIME_VERSION = "py3.13-aarch64-v1";
    private static final String PY_RUNTIME_VERSION_FILE = "py_runtime_version.txt";

    static File getPyRuntimeDir(Context context) {
        return new File(context.getFilesDir(), "py_runtime");
    }

    static String getPyRuntimeHome(Context context) {
        return new File(getPyRuntimeDir(context), "usr").getAbsolutePath();
    }

    static String getPyRuntimeLibDir(Context context) {
        return new File(getPyRuntimeDir(context), "usr/lib").getAbsolutePath();
    }

    /** Returns absolute path to py_runtime Python binary, or null if not yet extracted. */
    static String findBundledPyRuntimePython(Context context) {
        File py = new File(getPyRuntimeDir(context), "usr/bin/python3");
        return (py.isFile() && py.canExecute()) ? py.getAbsolutePath() : null;
    }

    /**
     * Ensures py_runtime.zip is extracted to files/py_runtime/.
     * Returns the directory path, or null if the asset is not bundled in this APK.
     */
    static String ensurePyRuntimeExtracted(Context context) throws IOException {
        // Check if asset exists in APK
        InputStream probe = null;
        try {
            probe = context.getAssets().open(PY_RUNTIME_ZIP_ASSET);
        } catch (IOException e) {
            return null; // not bundled
        } finally {
            if (probe != null) probe.close();
        }

        File pyRuntimeDir = getPyRuntimeDir(context);
        File versionFile = new File(pyRuntimeDir, PY_RUNTIME_VERSION_FILE);

        // Check if already extracted with matching version
        if (versionFile.isFile()) {
            try {
                String current = readTextFile(versionFile).trim();
                if (PY_RUNTIME_VERSION.equals(current)) {
                    String python = findBundledPyRuntimePython(context);
                    if (python != null) {
                        Log.i(TAG, "py_runtime already extracted: " + current);
                        return pyRuntimeDir.getAbsolutePath();
                    }
                }
            } catch (IOException ignored) {
            }
        }

        Log.i(TAG, "Extracting py_runtime.zip to " + pyRuntimeDir.getAbsolutePath());

        // Clean up old extraction
        deleteRecursively(pyRuntimeDir);
        if (!pyRuntimeDir.mkdirs() && !pyRuntimeDir.isDirectory()) {
            throw new IOException("Cannot create py_runtime dir: " + pyRuntimeDir);
        }

        // Extract zip
        byte[] buf = new byte[COPY_BUFFER_SIZE];
        try (InputStream assetIn = context.getAssets().open(PY_RUNTIME_ZIP_ASSET);
             ZipInputStream zis = new ZipInputStream(assetIn)) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.isDirectory()) {
                    new File(pyRuntimeDir, entry.getName()).mkdirs();
                } else {
                    File dest = new File(pyRuntimeDir, entry.getName());
                    File parent = dest.getParentFile();
                    if (parent != null && !parent.isDirectory()) parent.mkdirs();
                    try (FileOutputStream fos = new FileOutputStream(dest)) {
                        int n;
                        while ((n = zis.read(buf)) != -1) {
                            fos.write(buf, 0, n);
                        }
                    }
                }
                zis.closeEntry();
            }
        }

        // Set execute permissions on binaries and .so files
        setPyRuntimeExecutablePermissions(pyRuntimeDir);

        // Write version marker
        writeTextFile(versionFile, PY_RUNTIME_VERSION);

        Log.i(TAG, "py_runtime extracted successfully");
        return pyRuntimeDir.getAbsolutePath();
    }

    private static void setPyRuntimeExecutablePermissions(File pyRuntimeDir) {
        // Executables in usr/bin
        File binDir = new File(pyRuntimeDir, "usr/bin");
        if (binDir.isDirectory()) {
            File[] files = binDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile()) f.setExecutable(true, false);
                }
            }
        }
        // Shared libraries in usr/lib
        File libDir = new File(pyRuntimeDir, "usr/lib");
        if (libDir.isDirectory()) {
            File[] files = libDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile() && f.getName().contains(".so")) f.setExecutable(true, false);
                }
            }
        }
        // Extension modules in usr/lib/python3.13/lib-dynload
        File dynloadDir = new File(pyRuntimeDir, "usr/lib/python3.13/lib-dynload");
        if (dynloadDir.isDirectory()) {
            File[] files = dynloadDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile() && f.getName().endsWith(".so")) f.setExecutable(true, false);
                }
            }
        }
    }
}

