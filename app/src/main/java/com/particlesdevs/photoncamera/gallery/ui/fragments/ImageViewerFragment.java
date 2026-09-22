package com.particlesdevs.photoncamera.gallery.ui.fragments;

import android.app.Activity;
import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.TimeInterpolator;
import android.animation.ValueAnimator;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.PointF;
import android.graphics.RenderEffect;
import android.graphics.RuntimeShader;
import android.graphics.Shader;
import android.net.Uri;
import android.os.Bundle;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.MimeTypeMap;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.particlesdevs.photoncamera.util.BlurSupport;
import androidx.databinding.DataBindingUtil;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.fragment.NavHostFragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.transition.ChangeBounds;
import androidx.transition.TransitionManager;
import androidx.viewpager2.widget.ViewPager2;

import com.davemorrissey.labs.subscaleview.SubsamplingScaleImageView;
import com.particlesdevs.photoncamera.R;
import com.particlesdevs.photoncamera.app.PhotonCamera;
import com.particlesdevs.photoncamera.circularbarlib.util.Motion;
import com.particlesdevs.photoncamera.control.Vibration;
import com.particlesdevs.photoncamera.databinding.FragmentGalleryImageViewerBinding;
import com.particlesdevs.photoncamera.gallery.adapters.ImageAdapter;
import com.particlesdevs.photoncamera.gallery.adapters.ImageGridAdapter;
import com.particlesdevs.photoncamera.gallery.compare.SSIVListener;
import com.particlesdevs.photoncamera.gallery.files.GalleryFileOperations;
import com.particlesdevs.photoncamera.gallery.files.ImageFile;
import com.particlesdevs.photoncamera.gallery.helper.Constants;
import com.particlesdevs.photoncamera.gallery.helper.UltraHdrGalleryUtil;
import com.particlesdevs.photoncamera.gallery.model.GalleryItem;
import com.particlesdevs.photoncamera.gallery.viewmodel.ExifDialogViewModel;
import com.particlesdevs.photoncamera.gallery.viewmodel.GalleryViewModel;
import com.particlesdevs.photoncamera.gallery.views.CustomSSIV;
import com.particlesdevs.photoncamera.processing.ImagePath;
import com.particlesdevs.photoncamera.util.SystemBarsHelper;

import org.apache.commons.io.FileUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class ImageViewerFragment extends Fragment implements ImageAdapter.HdrStateListener {
    private List<GalleryItem> galleryItems=new ArrayList<>(0);
    private ExifDialogViewModel exifDialogViewModel;
    private ViewPager2 viewPager;
    private RecyclerView linearRecyclerView;
    private ImageAdapter adapter;
    private ImageGridAdapter linearGridAdapter;
    private NavController navController;
    private FragmentGalleryImageViewerBinding fragmentGalleryImageViewerBinding;
    private boolean isExifVisible;
    /** Whether the scrollable EXIF description inside the panel is expanded. */
    private boolean isDescriptionExpanded;
    /**
     * Clock animating alongside the panel's bounds toggle so the backdrop
     * snapshot can be recaptured while the panel grows or shrinks.
     */
    private ValueAnimator descriptionBlurClock;
    /** Blur radius applied to the EXIF panel backdrop, in dp. */
    private static final float EXIF_BLUR_RADIUS_DP = 32f;
    /** Downscale of the captured backdrop bitmap (keeps both capture and software blur cheap). */
    private static final float EXIF_BLUR_CAPTURE_SCALE = 4f;
    /** 40% dark scrim baked into the backdrop for text legibility. */
    private static final int EXIF_BLUR_SCRIM = 0x66121417;
    /** Minimum interval between backdrop refreshes while panning/zooming. */
    private static final long EXIF_BLUR_THROTTLE_MS = 80L;
    /**
     * Rounded-corner mask applied *after* the GPU blur. Only RenderEffect chains
     * can mask after a blur without smearing the content back into the corners.
     */
    private static final String EXIF_MASK_AGSL =
            "uniform shader content;\n" +
            "uniform float2 size;\n" +
            "uniform float radius;\n" +
            "half4 main(float2 coord) {\n" +
            "    float2 halfSize = size * 0.5;\n" +
            "    float2 d = abs(coord - halfSize) - max(halfSize - radius, float2(0.0));\n" +
            "    float dist = length(max(d, float2(0.0))) + min(max(d.x, d.y), 0.0) - radius;\n" +
            "    float mask = 1.0 - smoothstep(-1.0, 1.0, dist);\n" +
            "    half4 c = content.eval(coord);\n" +
            "    return half4(c.rgb * mask, c.a * mask);\n" +
            "}\n";

    private final Bitmap[] exifBlurBuffers = new Bitmap[2];
    private int exifBlurBufferIndex;
    private int exifBlurLayoutRetries;
    private RuntimeShader exifMaskShader;
    private RenderEffect exifBackdropEffect;
    private int exifEffectWidth;
    private int exifEffectHeight;
    private long exifBlurLastCaptureMs;
    private final Handler exifBlurHandler = new Handler(Looper.getMainLooper());
    private final Runnable exifBlurCaptureRunnable = this::captureExifBlur;
    private String mode;
    private int seek_position = 0;
    private int lastHdrPosition = -1;
    private int deferredReleasePos = -1;
    private SSIVListener ssivListener = new SSIVListener() {
        @Override public void onScaleChanged(float newScale, int origin) {
            updateScaleText();
            if (origin == SubsamplingScaleImageView.ORIGIN_DOUBLE_TAP_ZOOM && vibration != null) {
                vibration.zoomDetent();
            }
            if (viewPager != null) {
                CustomSSIV cur = getCurrentSSIV();
                if (cur != null && cur.isReady()) {
                    boolean zoomed = cur.getScale() > cur.getMinScale() + 0.02f;
                    viewPager.setUserInputEnabled(!zoomed);
                } else if (viewPager != null) {
                    viewPager.setUserInputEnabled(true);
                }
            }
            scheduleExifBlurRefresh();
        }
        @Override public void onCenterChanged(PointF newCenter, int origin) {
            scheduleExifBlurRefresh();
        }
        @Override public void onTouched(int id) {}
    };
    private int indexToDelete = -1;
    /**
     * Gallery chrome (top/bottom controls) state for photo pages. Video pages
     * always hide it so only the player controls show; swiping back to a
     * photo restores this value.
     */
    private boolean galleryChromeVisible;
    private GalleryViewModel viewModel;
    private Vibration vibration;
    private ViewPager2.OnPageChangeCallback pageCallback;
    // Deferred EXIF refresh after swipes: must be cancellable so it never
    // fires on a detached fragment (requireContext() would throw).
    private final Runnable exifUpdateRunnable = this::updateExif;

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Bundle args = getArguments();
        if (args != null) {
            mode = args.getString(Constants.MODE_KEY);
            seek_position = args.getInt(Constants.IMAGE_POSITION_KEY, 0);
        }
        if (savedInstanceState != null) {
            seek_position = savedInstanceState.getInt(Constants.IMAGE_POSITION_KEY, seek_position);
        }
    }

    @Override
    public void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        if (viewPager != null) outState.putInt(Constants.IMAGE_POSITION_KEY, viewPager.getCurrentItem());
        else outState.putInt(Constants.IMAGE_POSITION_KEY, seek_position);
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        fragmentGalleryImageViewerBinding = DataBindingUtil.inflate(inflater, R.layout.fragment_gallery_image_viewer, container, false);
        viewModel = new ViewModelProvider(requireActivity()).get(GalleryViewModel.class);
        initialiseDataMembers();
        if (fragmentGalleryImageViewerBinding.hdrToggleText != null) {
            fragmentGalleryImageViewerBinding.hdrToggleText.setOnClickListener(this::onHdrToggleClicked);
        }
        setClickListeners();
        return fragmentGalleryImageViewerBinding.getRoot();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // The deferred EXIF runnable must not outlive the view: it touches
        // requireContext() and would crash on a detached fragment.
        if (viewPager != null) {
            viewPager.removeCallbacks(exifUpdateRunnable);
        }
        if (adapter != null) adapter.releaseVideoPlayer();
        if (descriptionBlurClock != null) {
            ValueAnimator clock = descriptionBlurClock;
            descriptionBlurClock = null;
            clock.cancel();
        }
        exifBlurHandler.removeCallbacks(exifBlurCaptureRunnable);
        clearExifBlur();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (viewPager != null && pageCallback != null) {
            viewPager.unregisterOnPageChangeCallback(pageCallback);
        }
        if (adapter != null) adapter.clearPreviewCaches();
        getParentFragmentManager().beginTransaction().remove((Fragment) ImageViewerFragment.this).commitAllowingStateLoss();
        fragmentGalleryImageViewerBinding = null;
    }

    private void initialiseDataMembers() {
        viewPager = fragmentGalleryImageViewerBinding.viewPager;
        linearRecyclerView = fragmentGalleryImageViewerBinding.bottomControlsContainer.scrollingGalleryView;
        exifDialogViewModel = new ViewModelProvider(this).get(ExifDialogViewModel.class);
        fragmentGalleryImageViewerBinding.exifLayout.setExifmodel(exifDialogViewModel.getExifDataModel());
        fragmentGalleryImageViewerBinding.setExifmodel(exifDialogViewModel.getExifDataModel());
        navController = NavHostFragment.findNavController(this);
        viewModel.getCurrentFolderImages().observe(getViewLifecycleOwner(),this::initImageAdapter);
    }

    private void initImageAdapter(List<GalleryItem> galleryItems) {
        if (galleryItems != null) {
            this.galleryItems = galleryItems;
            if (seek_position >= galleryItems.size()) seek_position = Math.max(0, galleryItems.size()-1);
            if (seek_position < 0) seek_position = 0;
            adapter = new ImageAdapter(this.galleryItems);
            adapter.setAppContext(requireContext().getApplicationContext());
            adapter.setImageViewClickListener(ImageViewerFragment.this::onImageViewClicked);
            adapter.setHdrStateListener(this);
            if (ssivListener != null) adapter.setSsivListener(ssivListener);
            adapter.setImageEventListener(new SubsamplingScaleImageView.DefaultOnImageEventListener() {
                @Override public void onReady() {
                    updateScaleText();
                    scheduleExifBlurRefresh();
                }
            });
            viewPager.setAdapter(adapter);
            initLinearRecyclerAdapter(galleryItems);
            viewPager.setCurrentItem(seek_position, false);
            viewPager.post(() -> updateChromeForPosition(seek_position));
            // Eagerly preload previews for the window so neighbor pages show a placeholder (no black).
            adapter.preloadPreviews(seek_position);
            linearRecyclerView.post(() -> linearRecyclerView.scrollToPosition(seek_position));
            viewPager.post(() -> onPageHdrSelected(seek_position));
        }
    }

    private void initLinearRecyclerAdapter(List<GalleryItem> galleryItems) {
        if (galleryItems != null) {
            linearGridAdapter = new ImageGridAdapter(galleryItems, Constants.GALLERY_ITEM_TYPE_LINEAR);
            fragmentGalleryImageViewerBinding.bottomControlsContainer.scrollingGalleryView.setAdapter(linearGridAdapter);
            linearGridAdapter.setGridAdapterCallback(new ImageGridAdapter.GridAdapterCallback() {
                @Override public void onItemClicked(int position, View view, GalleryItem galleryItem) {
                    seek_position = position;
                    viewPager.setCurrentItem(position, true);
                    LinearLayoutManager lm = (LinearLayoutManager) linearRecyclerView.getLayoutManager();
                    if (lm != null) {
                        int avg = (lm.findFirstCompletelyVisibleItemPosition() + (lm.findFirstCompletelyVisibleItemPosition() + 1) +
                                lm.findLastCompletelyVisibleItemPosition()) / 3;
                        if (position > avg) linearRecyclerView.smoothScrollToPosition(position + 1);
                        else if (position != 0) linearRecyclerView.smoothScrollToPosition(position - 1);
                        else linearRecyclerView.smoothScrollToPosition(0);
                    }
                }
                @Override public void onImageSelectionChanged(int num) {}
                @Override public void onImageSelectionStopped() {}
            });
        }
    }

    private void setClickListeners() {
        fragmentGalleryImageViewerBinding.bottomControlsContainer.setOnShare(this::onShareButtonClick);
        fragmentGalleryImageViewerBinding.bottomControlsContainer.setOnDelete(this::onDeleteButtonClick);
        fragmentGalleryImageViewerBinding.bottomControlsContainer.setOnExif(this::onExifButtonClick);
        fragmentGalleryImageViewerBinding.bottomControlsContainer.setOnShare(this::onShareButtonClick);
        fragmentGalleryImageViewerBinding.bottomControlsContainer.setOnEdit(this::onEditButtonClick);
        fragmentGalleryImageViewerBinding.topControlsContainer.setOnGallery(this::onGalleryButtonClick);
        fragmentGalleryImageViewerBinding.topControlsContainer.setOnBack(this::onBack);
        fragmentGalleryImageViewerBinding.topControlsContainer.setOnQuickCompare(this::onQuickCompare);
        fragmentGalleryImageViewerBinding.exifLayout.histogramView.setHistogramLoadingListener(this::isHistogramLoading);
        fragmentGalleryImageViewerBinding.exifLayout.exifDescriptionToggle.setOnClickListener(this::onDescriptionToggleClicked);
        fragmentGalleryImageViewerBinding.setOnclickempty(this::onEmptyViewClicked);
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        vibration = PhotonCamera.getVibration();
        // Keep bottom controls above the transparent navigation bar.
        // The photo pager itself stays full-bleed behind it.
        View bottomControls = view.findViewById(R.id.bottom_controls_container);
        if (bottomControls != null) {
            SystemBarsHelper.padBottomForNavBar(bottomControls);
        }
        View miniExif = view.findViewById(R.id.mini_exif_container);
        if (miniExif != null) {
            SystemBarsHelper.padBottomForNavBar(miniExif);
        }
        viewPager.setPageTransformer(null);
        // D: O(viewport) ~10-12 MB per page at 160dpi; keep baseline low on 4GB: offscreen=1 and cache=1 (was 2/2).
        // Still pre-warms ±1 tile via preview placeholder, without extra native tile retention.
        viewPager.setOffscreenPageLimit(1);
        viewPager.setUserInputEnabled(true);
        try {
            RecyclerView rv = (RecyclerView) viewPager.getChildAt(0);
            if (rv != null) {
                rv.setHasFixedSize(true);
                rv.setItemViewCacheSize(1);
            }
        } catch (Exception ignored) {}
        pageCallback = new ViewPager2.OnPageChangeCallback() {
            @Override public void onPageSelected(int position) {
                seek_position = position;
                if (vibration != null) vibration.pageSnap();
                updateChromeForPosition(position);
                boolean isVideo = adapter != null && adapter.isVideoPosition(position);
                if (isVideo) {
                    resetScaleText();
                    if (adapter != null) adapter.playVideoAt(position);
                } else {
                    if (adapter != null) adapter.stopVideo();
                    updateScaleText();
                }
                linearRecyclerView.smoothScrollToPosition(position);
                onPageHdrSelected(position);
                viewPager.setUserInputEnabled(true);
                // Re-arm preview preload for the new window so neighbors remain instant.
                if (adapter != null) adapter.preloadPreviews(position);
                // Defer heavy Exif/Histogram off critical swipe jank (saves ~48ms).
                // Re-post (not pile up) so only the settled page refreshes.
                viewPager.removeCallbacks(exifUpdateRunnable);
                viewPager.postDelayed(exifUpdateRunnable, 120);
            }
            @Override public void onPageScrollStateChanged(int state) {
                if (state == ViewPager2.SCROLL_STATE_IDLE && deferredReleasePos != -1) {
                    CustomSSIV prev = adapter != null ? adapter.getActiveSsiv(deferredReleasePos) : null;
                    if (prev == null) prev = getSsivAt(deferredReleasePos);
                    if (adapter != null) adapter.releaseHdrForPosition(prev, deferredReleasePos);
                    deferredReleasePos = -1;
                    updateWindowHdrForVisible();
                }
            }
        };
        viewPager.registerOnPageChangeCallback(pageCallback);
        updateExif();
    }

    @Override public void onResume() {
        super.onResume();
        if (fragmentGalleryImageViewerBinding != null && viewPager != null)
            updateChromeForPosition(viewPager.getCurrentItem());
        if (adapter != null && viewPager != null
                && isVideoPosition(viewPager.getCurrentItem())) {
            adapter.playVideoAt(viewPager.getCurrentItem());
        }
        if (adapter != null && viewPager != null) {
            int position = viewPager.getCurrentItem();
            seek_position = position;
            if (adapter.isHdrActive(position)) {
                updateWindowHdrForVisible();
                updateHdrToggleUi(adapter.isHdrAvailable(position), true);
            } else {
                CustomSSIV ssiv = getSsivAt(position);
                if (ssiv != null) adapter.loadHdrForPosition(ssiv, position);
                else viewPager.post(() -> adapter.loadHdrForPosition(getSsivAt(position), position));
                updateHdrToggleUi(adapter.isHdrAvailable(position), false);
                updateWindowHdrForVisible();
            }
        }
    }

    @Override public void onPause() {
        super.onPause();
        if (viewPager != null) seek_position = viewPager.getCurrentItem();
        if (adapter != null) adapter.pauseVideo();
        UltraHdrGalleryUtil.setWindowHdr(getActivity(), false);
        if (adapter != null && viewPager != null) {
            int position = viewPager.getCurrentItem();
            updateHdrToggleUi(adapter.isHdrAvailable(position), false);
        }
        if (viewPager != null) viewPager.setUserInputEnabled(true);
        if (deferredReleasePos != -1 && adapter != null) {
            CustomSSIV prev = adapter.getActiveSsiv(deferredReleasePos);
            if (prev == null) prev = getSsivAt(deferredReleasePos);
            adapter.releaseHdrForPosition(prev, deferredReleasePos);
            deferredReleasePos = -1;
        }
        // C: trim caches when backgrounded to drop baseline quickly on 4GB
        if (adapter != null) adapter.trimCaches(android.content.ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN);
    }

    private void onPageHdrSelected(int position) {
        if (adapter == null) return;
        if (lastHdrPosition >= 0 && lastHdrPosition != position) {
            deferredReleasePos = lastHdrPosition;
        }
        lastHdrPosition = position;
        CustomSSIV cur = adapter.getActiveSsiv(position);
        if (cur == null) cur = getSsivAt(position);
        if (cur != null) adapter.loadHdrForPosition(cur, position);
        else viewPager.post(() -> adapter.loadHdrForPosition(getSsivAt(position), position));
        updateWindowHdrForVisible();
    }

    private void updateWindowHdrForVisible() {
        if (getActivity() == null || adapter == null || viewPager == null) return;
        if (isCompareMode()) {
            UltraHdrGalleryUtil.setWindowHdr(getActivity(), adapter.isHdrActive(viewPager.getCurrentItem()));
            return;
        }
        boolean needHdr = adapter.isHdrActive(viewPager.getCurrentItem());
        if (!needHdr && deferredReleasePos != -1) needHdr = adapter.isHdrActive(deferredReleasePos);
        UltraHdrGalleryUtil.setWindowHdr(getActivity(), needHdr);
    }

    @Override public void onHdrStateChanged(int position, boolean isHdr) {
        if (getActivity() != null && viewPager != null && position == viewPager.getCurrentItem()) {
            updateWindowHdrForVisible();
            updateHdrToggleUi(adapter != null && adapter.isHdrAvailable(position), isHdr);
        }
    }

    private void onHdrToggleClicked(View view) {
        if (adapter == null || viewPager == null) return;
        int position = viewPager.getCurrentItem();
        if (!adapter.isHdrAvailable(position)) return;
        if (vibration != null) vibration.toggle(!adapter.isHdrActive(position));
        CustomSSIV ssiv = getSsivAt(position);
        if (adapter.isHdrActive(position)) {
            adapter.releaseHdrForPosition(ssiv, position);
            UltraHdrGalleryUtil.setWindowHdr(getActivity(), false);
            updateHdrToggleUi(true, false);
        } else {
            adapter.loadHdrForPosition(ssiv, position);
        }
    }

    @Override public void onHdrAvailabilityChanged(int position, boolean isUltraHdr) {
        if (viewPager != null && position == viewPager.getCurrentItem()) {
            boolean isHdr = adapter != null && adapter.isHdrActive(position);
            updateHdrToggleUi(isUltraHdr, isHdr);
        }
    }

    private void updateHdrToggleUi(boolean isUltraHdr, boolean isHdr) {
        if (fragmentGalleryImageViewerBinding == null || fragmentGalleryImageViewerBinding.hdrToggleText == null) return;
        if (!isUltraHdr || !fragmentGalleryImageViewerBinding.getButtonsVisible()) {
            fragmentGalleryImageViewerBinding.hdrToggleText.setVisibility(View.GONE);
            return;
        }
        fragmentGalleryImageViewerBinding.hdrToggleText.setVisibility(View.VISIBLE);
        fragmentGalleryImageViewerBinding.hdrToggleText.setImageResource(isHdr ? R.drawable.ic_ultra_hdr : R.drawable.ic_ultra_hdr_off);
    }

    public void setSsivListener(SSIVListener ssivListener) { this.ssivListener = ssivListener; }
    public CustomSSIV getCurrentSSIV() { return getSsivAt(viewPager != null ? viewPager.getCurrentItem() : seek_position); }

    private CustomSSIV getSsivAt(int position) {
        if (adapter == null || viewPager == null) return null;
        CustomSSIV fromMap = adapter.getActiveSsiv(position);
        if (fromMap != null) return fromMap;
        try {
            RecyclerView rv = (RecyclerView) viewPager.getChildAt(0);
            if (rv != null) {
                RecyclerView.ViewHolder vh = rv.findViewHolderForAdapterPosition(position);
                if (vh instanceof ImageAdapter.Holder) return ((ImageAdapter.Holder) vh).getSsiv();
                for (int i = 0; i < rv.getChildCount(); i++) {
                    View child = rv.getChildAt(i);
                    RecyclerView.ViewHolder ch = rv.getChildViewHolder(child);
                    if (ch != null && ch.getBindingAdapterPosition() == position && ch instanceof ImageAdapter.Holder) {
                        return ((ImageAdapter.Holder) ch).getSsiv();
                    }
                }
            }
        } catch (Exception ignored) {}
        return null;
    }

    private void onBack(View view) {
        if (vibration != null) vibration.confirm();
        // Match system back: pop the nav graph; only finish when launched externally.
        if (navController != null && navController.navigateUp()) return;
        if (getActivity() != null) getActivity().finish();
    }

    private boolean isVideoPosition(int position) {
        return galleryItems != null && position >= 0 && position < galleryItems.size()
                && galleryItems.get(position).isVideo();
    }

    private void onQuickCompare(View view) {
        if (galleryItems.size() >= 2) {
            int image1pos = viewPager.getCurrentItem();
            int image2pos = image1pos + 1;
            if (image1pos == galleryItems.size() - 1) { image2pos = image1pos; image1pos -= 1; }
            if (isVideoPosition(image1pos) || isVideoPosition(image2pos)) {
                Toast.makeText(getContext(), "Compare is available for photos only", Toast.LENGTH_SHORT).show();
                return;
            }
            if (vibration != null) vibration.confirm();
            NavController navController = Navigation.findNavController(view);
            Bundle b = new Bundle(2);
            b.putInt(Constants.IMAGE1_KEY, image1pos);
            b.putInt(Constants.IMAGE2_KEY, image2pos);
            navController.navigate(R.id.action_imageViewerFragment_to_imageCompareFragment, b);
        } else {
            if (vibration != null) vibration.reject();
            Toast.makeText(getContext(), "No images to compare!", Toast.LENGTH_SHORT).show();
        }
    }

    private void onGalleryButtonClick(View view) {
        if (vibration != null) vibration.confirm();
        if (navController.getPreviousBackStackEntry() == null)
            navController.navigate(R.id.action_imageViewFragment_to_imageLibraryFragment);
        else navController.navigateUp();
    }

    private void onEditButtonClick(View view) {
        if (vibration != null) vibration.confirm();
        int position = viewPager.getCurrentItem();
        if (isVideoPosition(position)) {
            Toast.makeText(getContext(), "Video editing is not supported", Toast.LENGTH_SHORT).show();
            return;
        }
        if (galleryItems != null && getContext() != null) {
            GalleryItem galleryItem = galleryItems.get(position);
            String fileName = galleryItem.getFile().getDisplayName();
            String mediaType = MimeTypeMap.getSingleton().getMimeTypeFromExtension(FileUtils.getExtension(fileName));
            Uri uri = galleryItem.getFile().getFileUri();
            Intent editIntent = new Intent(Intent.ACTION_EDIT);
            editIntent.setDataAndType(uri, mediaType);
            String outPutFileUri = galleryItem.getFile().getFileUri().toString().replace(galleryItem.getFile().getDisplayName(), ImagePath.generateNewFileName("IMG") + '.' + FileUtils.getExtension(fileName));
            editIntent.putExtra(MediaStore.EXTRA_OUTPUT, outPutFileUri);
            editIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            Intent chooser = Intent.createChooser(editIntent, null);
            startActivityForResult(chooser, Constants.REQUEST_EDIT_IMAGE);
        }
    }

    @Override public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == Constants.REQUEST_EDIT_IMAGE) {
            if (resultCode == Activity.RESULT_OK && data != null && data.getData() != null) {
                if (vibration != null) vibration.confirm();
                String savedFilePath = data.getData().getPath();
                Toast.makeText(getContext(), "Saved : " + savedFilePath, Toast.LENGTH_LONG).show();
                viewModel.fetchAllMedia();
                initImageAdapter(viewModel.getCurrentFolderImages().getValue());
                refreshLinearGridAdapter(viewModel.getCurrentFolderImages().getValue());
                updateExif();
            }
        }
    }

    private void refreshLinearGridAdapter(List<GalleryItem> galleryItems) {
        linearGridAdapter.setGalleryItemList(galleryItems);
        linearGridAdapter.notifyDataSetChanged();
    }

    private void onDeleteButtonClick(View view) {
        MaterialAlertDialogBuilder builder = new MaterialAlertDialogBuilder(getContext());
        builder.setMessage(R.string.sure_delete).setTitle(android.R.string.dialog_alert_title).setIcon(R.drawable.ic_delete).setNegativeButton(R.string.cancel, (dialog, which) -> dialog.dismiss())
                .setPositiveButton(R.string.yes, (dialog, which) -> {
                    if (vibration != null) vibration.confirm();
                    indexToDelete = viewPager.getCurrentItem();
                    GalleryFileOperations.deleteImageFiles(getActivity(), Collections.singletonList((ImageFile) galleryItems.get(indexToDelete).getFile()), this::handleImagesDeletedCallback);
                });
        BlurSupport.show(builder.create());
    }

    private void onShareButtonClick(View view) {
        if (vibration != null) vibration.confirm();
        int position = viewPager.getCurrentItem();
        GalleryItem galleryItem = galleryItems.get(position);
        String fileName = galleryItem.getFile().getDisplayName();
        String mediaType = MimeTypeMap.getSingleton().getMimeTypeFromExtension(FileUtils.getExtension(fileName));
        Uri uri = galleryItem.getFile().getFileUri();
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.putExtra(Intent.EXTRA_STREAM, uri);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.setType(mediaType);
        startActivity(Intent.createChooser(intent, null));
    }

    private void onExifButtonClick(View view) {
        isExifVisible = !isExifVisible;
        if (vibration != null) vibration.toggle(isExifVisible);
        fragmentGalleryImageViewerBinding.setExifDialogVisible(isExifVisible);
        updateExif();
    }

    private void onDescriptionToggleClicked(View view) {
        if (fragmentGalleryImageViewerBinding == null || fragmentGalleryImageViewerBinding.exifLayout == null) return;
        View panel = fragmentGalleryImageViewerBinding.exifLayout.getRoot();
        View scroll = fragmentGalleryImageViewerBinding.exifLayout.exifDescriptionScroll;
        if (panel == null || scroll == null) return;
        Context context = view.getContext();
        isDescriptionExpanded = !isDescriptionExpanded;
        if (vibration != null) vibration.toggle(isDescriptionExpanded);
        int duration = Motion.durationMedium1(context);
        TimeInterpolator interpolator = Motion.emphasized(context);
        // Animate the panel's bounds so the description is revealed downwards;
        // ChangeBounds interpolates the panel and every child below the toggle.
        ViewGroup parent = panel.getParent() instanceof ViewGroup ? (ViewGroup) panel.getParent() : null;
        TransitionManager.beginDelayedTransition(parent != null ? parent : (ViewGroup) panel,
                new ChangeBounds().setDuration(duration).setInterpolator(interpolator));
        scroll.setVisibility(isDescriptionExpanded ? View.VISIBLE : View.GONE);
        startDescriptionBlurClock(panel, duration, interpolator);
        updateDescriptionToggleUi();
    }

    /**
     * The backdrop is a snapshot of the image behind the panel, so it has to be
     * recaptured at the panel's current, animated size every frame -- otherwise
     * the stale capture only stretches over the new area and the blur visibly
     * lands after the text. A clock with the transition's duration/interpolator
     * keeps the captures aligned with the bounds animation, then posts one last
     * capture a frame after the panel settles.
     */
    private void startDescriptionBlurClock(View panel, int duration, TimeInterpolator interpolator) {
        if (descriptionBlurClock != null) {
            descriptionBlurClock.cancel();
        }
        ValueAnimator clock = ValueAnimator.ofFloat(0f, 1f).setDuration(duration);
        clock.setInterpolator(interpolator);
        clock.addUpdateListener(animation -> captureExifBlur());
        clock.addListener(new AnimatorListenerAdapter() {
            @Override public void onAnimationEnd(Animator animation) {
                boolean finished = descriptionBlurClock == clock;
                if (finished) descriptionBlurClock = null;
                if (finished && getView() != null) {
                    panel.post(ImageViewerFragment.this::captureExifBlur);
                }
            }
        });
        descriptionBlurClock = clock;
        clock.start();
    }

    /** Mirrors the model's description availability and the expanded state onto the panel. */
    private void syncDescriptionToggle() {
        if (fragmentGalleryImageViewerBinding == null || fragmentGalleryImageViewerBinding.exifLayout == null) return;
        View scroll = fragmentGalleryImageViewerBinding.exifLayout.exifDescriptionScroll;
        ImageView toggle = fragmentGalleryImageViewerBinding.exifLayout.exifDescriptionToggle;
        if (scroll == null || toggle == null) return;
        String description = exifDialogViewModel.getExifDataModel().getDescription();
        boolean available = description != null && !description.isEmpty();
        if (!available) isDescriptionExpanded = false;
        scroll.setVisibility(available && isDescriptionExpanded ? View.VISIBLE : View.GONE);
        toggle.setVisibility(available ? View.VISIBLE : View.GONE);
        updateDescriptionToggleUi();
    }

    private void updateDescriptionToggleUi() {
        if (fragmentGalleryImageViewerBinding == null || fragmentGalleryImageViewerBinding.exifLayout == null) return;
        ImageView toggle = fragmentGalleryImageViewerBinding.exifLayout.exifDescriptionToggle;
        if (toggle == null) return;
        Context context = toggle.getContext();
        toggle.animate()
                .rotation(isDescriptionExpanded ? 0f : 180f)
                .setDuration(Motion.durationShort4(context))
                .setInterpolator(Motion.emphasized(context))
                .start();
        toggle.setContentDescription(context.getString(isDescriptionExpanded
                ? R.string.exif_hide_description : R.string.exif_show_description));
    }

    private void onImageViewClicked(View view) {
        if (vibration != null) vibration.chromeToggle();
        if (isCompareMode()) {
            onExifButtonClick(null);
            fragmentGalleryImageViewerBinding.setMiniExifVisible(!isExifVisible);
        } else {
            fragmentGalleryImageViewerBinding.setButtonsVisible(!fragmentGalleryImageViewerBinding.getButtonsVisible());
            galleryChromeVisible = fragmentGalleryImageViewerBinding.getButtonsVisible();
            int position = viewPager.getCurrentItem();
            updateHdrToggleUi(adapter != null && adapter.isHdrAvailable(position), adapter != null && adapter.isHdrActive(position));
            fragmentGalleryImageViewerBinding.setMiniExifVisible(!fragmentGalleryImageViewerBinding.getButtonsVisible());
            if (isExifVisible) {
                fragmentGalleryImageViewerBinding.setExifDialogVisible(fragmentGalleryImageViewerBinding.getButtonsVisible());
                updateExif();
            }
        }
    }

    /**
     * Shows gallery chrome only on photo pages; video pages hide it entirely
     * so just the player controls are visible. Photo pages restore the last
     * chrome state the user left them in.
     */
    private void updateChromeForPosition(int position) {
        if (fragmentGalleryImageViewerBinding == null) return;
        if (isVideoPosition(position)) {
            isExifVisible = false;
            fragmentGalleryImageViewerBinding.setExifDialogVisible(false);
            fragmentGalleryImageViewerBinding.setButtonsVisible(false);
            fragmentGalleryImageViewerBinding.setMiniExifVisible(false);
            resetScaleText();
            updateHdrToggleUi(false, false);
        } else {
            fragmentGalleryImageViewerBinding.setButtonsVisible(galleryChromeVisible);
            fragmentGalleryImageViewerBinding.setMiniExifVisible(!galleryChromeVisible);
        }
    }

    private void onEmptyViewClicked(View view) {
        NavController navController = Navigation.findNavController(view);
        navController.navigate(R.id.action_imageViewerFragment_to_gallerySettingsFragment);
    }

    public void updateScaleText() {
        SubsamplingScaleImageView view = getCurrentSSIV();
        if (view != null && fragmentGalleryImageViewerBinding != null) {
            fragmentGalleryImageViewerBinding.setScale(String.format(Locale.ROOT, "%.0f%%", (view.getScale() * 100)));
        }
    }

    public void resetScaleText() { if (fragmentGalleryImageViewerBinding!=null) fragmentGalleryImageViewerBinding.setScale(""); }

    private void updateExif() {
        // May run from a delayed post after navigation; never touch
        // requireContext() when detached.
        if (!isAdded() || viewPager == null) return;
        int position = viewPager.getCurrentItem();
        if (galleryItems != null && !galleryItems.isEmpty() && position < galleryItems.size()) {
            GalleryItem galleryItem = galleryItems.get(position);
            if (galleryItem.isVideo()) {
                updateVideoModel(galleryItem);
            } else {
                exifDialogViewModel.updateModel(requireContext().getContentResolver(), galleryItem.getFile());
                if (fragmentGalleryImageViewerBinding.getExifDialogVisible()) {
                    exifDialogViewModel.updateHistogramView((ImageFile) galleryItem.getFile());
                }
            }
        }
        syncDescriptionToggle();
        syncExifBlur();
    }

    /**
     * EXIF panel content for videos (no EXIF tags to read): file identity,
     * size and duration instead of exposure metadata.
     */
    private void updateVideoModel(GalleryItem galleryItem) {
        if (exifDialogViewModel == null || galleryItem.getFile() == null) return;
        com.particlesdevs.photoncamera.gallery.model.ExifDialogModel model =
                exifDialogViewModel.getExifDataModel();
        String duration = "";
        try {
            android.media.MediaMetadataRetriever retriever = new android.media.MediaMetadataRetriever();
            try {
                retriever.setDataSource(requireContext(), galleryItem.getFile().getFileUri());
                String ms = retriever.extractMetadata(
                        android.media.MediaMetadataRetriever.METADATA_KEY_DURATION);
                if (ms != null) {
                    long totalSeconds = Long.parseLong(ms) / 1000;
                    duration = String.format(java.util.Locale.US, "%02d:%02d",
                            totalSeconds / 60, totalSeconds % 60);
                }
                String w = retriever.extractMetadata(
                        android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);
                String h = retriever.extractMetadata(
                        android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);
                if (w != null && h != null) {
                    model.setRes(h + "x" + w);
                    try {
                        double mp = Double.parseDouble(w) * Double.parseDouble(h) / 1E6;
                        model.setRes_mp(String.format(java.util.Locale.US, "%.1f MP", mp));
                    } catch (Exception ignored) {
                        model.setRes_mp("");
                    }
                } else {
                    model.setRes("");
                    model.setRes_mp("");
                }
            } finally {
                try {
                    retriever.release();
                } catch (Exception ignored) {}
            }
        } catch (Exception ignored) {
            model.setRes("");
            model.setRes_mp("");
        }
        model.setTitle(galleryItem.getFile().getAbsolutePath());
        model.setDevice("");
        model.setDate("");
        model.setExposure(duration);
        model.setIso("");
        model.setFnum("");
        model.setFocal("");
        try {
            model.setFile_size(org.apache.commons.io.FileUtils.byteCountToDisplaySize(
                    (int) galleryItem.getFile().getSize()));
        } catch (Exception ignored) {
            model.setFile_size("");
        }
        model.setDescription("");
        String mini = galleryItem.getFile().getDisplayName() + "\nVideo"
                + (duration.isEmpty() ? "" : " | " + duration);
        model.setMiniText(mini);
        model.notifyChange();
    }

    /**
     * Shows, refreshes or clears the frosted-glass backdrop behind the EXIF
     * panel. Only the region behind the panel is captured and blurred; the
     * backdrop is clipped to the panel's rounded corners.
     */
    private void syncExifBlur() {
        if (!isAdded() || fragmentGalleryImageViewerBinding == null) {
            return;
        }
        boolean panelVisible = Boolean.TRUE.equals(fragmentGalleryImageViewerBinding.getExifDialogVisible());
        if (!panelVisible || Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            // Below API 33 there is no mask-after-blur; keep the opaque rounded panel.
            clearExifBlur();
            return;
        }
        // Show the scrim right away: the first blurred frame lands a few ms later
        // and the opaque panel must not flash in between.
        if (fragmentGalleryImageViewerBinding.exifLayout != null) {
            ImageView backdrop = fragmentGalleryImageViewerBinding.exifLayout.exifBlurBackdrop;
            if (backdrop == null || backdrop.getVisibility() != View.VISIBLE) {
                View panel = fragmentGalleryImageViewerBinding.exifLayout.getRoot();
                if (panel != null) {
                    panel.setBackgroundResource(R.drawable.exif_background_scrim);
                }
            }
        }
        scheduleExifBlurRefresh();
    }

    /** Throttled backdrop refresh, used while the image is panned or zoomed. */
    private void scheduleExifBlurRefresh() {
        if (!isAdded() || fragmentGalleryImageViewerBinding == null) {
            return;
        }
        if (!Boolean.TRUE.equals(fragmentGalleryImageViewerBinding.getExifDialogVisible())) {
            return;
        }
        exifBlurLayoutRetries = 0;
        long elapsed = SystemClock.uptimeMillis() - exifBlurLastCaptureMs;
        exifBlurHandler.removeCallbacks(exifBlurCaptureRunnable);
        if (elapsed >= EXIF_BLUR_THROTTLE_MS) {
            exifBlurHandler.post(exifBlurCaptureRunnable);
        } else {
            exifBlurHandler.postDelayed(exifBlurCaptureRunnable, EXIF_BLUR_THROTTLE_MS - elapsed);
        }
    }

    /**
     * Captures the visible part of the current page that sits behind the panel,
     * bakes in the 40% scrim and shows it blurred as the panel background.
     */
    private void captureExifBlur() {
        if (!isAdded() || fragmentGalleryImageViewerBinding == null) {
            return;
        }
        exifBlurLastCaptureMs = SystemClock.uptimeMillis();
        if (!Boolean.TRUE.equals(fragmentGalleryImageViewerBinding.getExifDialogVisible())) {
            return;
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return;
        }
        CustomSSIV ssiv = getCurrentSSIV();
        View panel = fragmentGalleryImageViewerBinding.exifLayout.getRoot();
        ImageView backdrop = fragmentGalleryImageViewerBinding.exifLayout.exifBlurBackdrop;
        if (ssiv == null || !ssiv.isReady() || panel == null || backdrop == null) {
            return;
        }
        int panelWidth = panel.getWidth();
        int panelHeight = panel.getHeight();
        if (panelWidth <= 0 || panelHeight <= 0) {
            // The panel is shown but not laid out yet: retry briefly on the next frames.
            if (exifBlurLayoutRetries++ < 10) {
                exifBlurHandler.postDelayed(exifBlurCaptureRunnable, 32L);
            }
            return;
        }
        exifBlurLayoutRetries = 0;
        int bitmapWidth = Math.max(1, Math.round(panelWidth / EXIF_BLUR_CAPTURE_SCALE));
        int bitmapHeight = Math.max(1, Math.round(panelHeight / EXIF_BLUR_CAPTURE_SCALE));
        // Ping-pong two buffers: the one currently shown is never written to,
        // so no draw can ever reference pixels we are modifying.
        int nextIndex = 1 - exifBlurBufferIndex;
        Bitmap capture = exifBlurBuffers[nextIndex];
        if (capture == null || capture.isRecycled()
                || capture.getWidth() != bitmapWidth || capture.getHeight() != bitmapHeight) {
            try {
                capture = Bitmap.createBitmap(bitmapWidth, bitmapHeight, Bitmap.Config.ARGB_8888);
            } catch (OutOfMemoryError e) {
                return;
            }
            exifBlurBuffers[nextIndex] = capture;
        }
        int[] panelLocation = new int[2];
        int[] ssivLocation = new int[2];
        panel.getLocationOnScreen(panelLocation);
        ssiv.getLocationOnScreen(ssivLocation);
        Canvas canvas = new Canvas(capture);
        canvas.scale(1f / EXIF_BLUR_CAPTURE_SCALE, 1f / EXIF_BLUR_CAPTURE_SCALE);
        canvas.translate(ssivLocation[0] - panelLocation[0], ssivLocation[1] - panelLocation[1]);
        ssiv.draw(canvas);
        // Only the scrim is baked here; the blur and rounded mask are applied by
        // the RenderEffect chain so corners stay crisp.
        canvas.drawColor(EXIF_BLUR_SCRIM);
        if (!prepareExifBackdropEffect(panelWidth, panelHeight)) {
            return;
        }
        // Apply the effect before showing the bitmap so the first visible frame
        // is already blurred (no sharp flash).
        backdrop.setRenderEffect(exifBackdropEffect);
        backdrop.setImageBitmap(capture);
        backdrop.setVisibility(View.VISIBLE);
        // Keep a rounded (transparent) background so the panel outline stays round.
        panel.setBackgroundResource(R.drawable.exif_background_transparent);
        exifBlurBufferIndex = nextIndex;
    }

    /**
     * Builds (or refreshes) the GPU effect that blurs the backdrop and then masks
     * it to the panel's rounded rectangle. Returns false when unavailable.
     */
    private boolean prepareExifBackdropEffect(int width, int height) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU || width <= 0 || height <= 0) {
            return false;
        }
        if (exifMaskShader == null) {
            exifMaskShader = new RuntimeShader(EXIF_MASK_AGSL);
        }
        if (exifBackdropEffect == null || exifEffectWidth != width || exifEffectHeight != height) {
            exifMaskShader.setFloatUniform("size", width, height);
            exifMaskShader.setFloatUniform("radius",
                    getResources().getDimension(R.dimen.cam_panel_corner_radius));
            float blurRadius = BlurSupport.dpToPx(requireContext(), EXIF_BLUR_RADIUS_DP);
            RenderEffect mask = RenderEffect.createRuntimeShaderEffect(exifMaskShader, "content");
            RenderEffect blur = RenderEffect.createBlurEffect(blurRadius, blurRadius, Shader.TileMode.CLAMP);
            exifBackdropEffect = RenderEffect.createChainEffect(mask, blur);
            exifEffectWidth = width;
            exifEffectHeight = height;
        }
        return true;
    }

    private void clearExifBlur() {
        exifBlurHandler.removeCallbacks(exifBlurCaptureRunnable);
        if (fragmentGalleryImageViewerBinding != null && fragmentGalleryImageViewerBinding.exifLayout != null) {
            ImageView backdrop = fragmentGalleryImageViewerBinding.exifLayout.exifBlurBackdrop;
            if (backdrop != null) {
                backdrop.setVisibility(View.GONE);
                backdrop.setImageBitmap(null);
                BlurSupport.clearBlur(backdrop);
            }
            View panel = fragmentGalleryImageViewerBinding.exifLayout.getRoot();
            if (panel != null) {
                panel.setBackgroundResource(R.drawable.exif_background);
            }
        }
        for (int i = 0; i < exifBlurBuffers.length; i++) {
            Bitmap buffer = exifBlurBuffers[i];
            if (buffer != null && !buffer.isRecycled()) {
                buffer.recycle();
            }
            exifBlurBuffers[i] = null;
        }
        exifBlurBufferIndex = 0;
    }

    private void isHistogramLoading(boolean loading) {
        new Handler(Looper.getMainLooper()).post(() -> {
            if (fragmentGalleryImageViewerBinding == null || fragmentGalleryImageViewerBinding.exifLayout == null) return;
            if (loading) fragmentGalleryImageViewerBinding.exifLayout.histoLoading.setVisibility(View.VISIBLE);
            else fragmentGalleryImageViewerBinding.exifLayout.histoLoading.setVisibility(View.INVISIBLE);
        });
    }

    private boolean isCompareMode() { return mode != null && mode.equalsIgnoreCase(Constants.COMPARE); }

    public void handleImagesDeletedCallback(boolean isDeleted) {
        if (!isAdded()) return;
        if (isDeleted && indexToDelete >= 0) {
            if (vibration != null) vibration.confirm();
            galleryItems.remove(indexToDelete);
            seek_position=indexToDelete;
            if (!galleryItems.isEmpty()) initImageAdapter(galleryItems);
            updateExif();
            Toast.makeText(getContext(), R.string.image_deleted, Toast.LENGTH_SHORT).show();
            indexToDelete = -1;
            if (galleryItems.isEmpty()) { viewModel.setUpdatePending(true); navController.navigateUp(); }
        } else {
            if (vibration != null) vibration.reject();
            Toast.makeText(getContext(), "Deletion Failed!", Toast.LENGTH_SHORT).show();
        }
    }
}
