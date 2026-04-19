package com.sdxlnpu.app;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;

final class RuntimeBootstrap {

    private static final String ASSET_ROOT = "termux_bundle";
    private static final String RUNTIME_PAYLOAD_DIR = "runtime_payload";
    private static final String VERSION_MARKER = ".bundle_version";
    private static final String BUNDLE_LAYOUT_VERSION = "termux-bundle-v2";
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
            return null;
        }

        File bundleDir = getBundleDir(context);
        String expectedVersion = BUNDLE_LAYOUT_VERSION;
        File marker = new File(bundleDir, VERSION_MARKER);
        if (bundleDir.isDirectory() && marker.isFile()) {
            String currentVersion = readTextFile(marker).trim();
            if (expectedVersion.equals(currentVersion)) {
                return bundleDir.getAbsolutePath();
            }
        }

        deleteRecursively(bundleDir);
        if (!bundleDir.mkdirs() && !bundleDir.isDirectory()) {
            throw new IOException("Не удалось создать каталог bundled runtime: " + bundleDir);
        }

        copyAssetTree(context.getAssets(), ASSET_ROOT, bundleDir);
        writeTextFile(marker, expectedVersion);

        // Set executable permissions on prefix/bin/* and prefix/lib/*.so
        setExecutablePermissions(bundleDir);

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

    private static void writeTextFile(File file, String content) throws IOException {
        try (OutputStreamWriter writer = new OutputStreamWriter(
                new FileOutputStream(file), StandardCharsets.UTF_8)) {
            writer.write(content);
        }
    }
}
