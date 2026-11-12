/**
 * ExtendScript Helper: Smooth Bezier Keyframe Interpolation
 * Converts selected clip property keyframes from linear to smooth Bezier.
 */

function setKeyframeBezier(trackIndex, clipIndex, propertyName) {
    try {
        var seq = app.project.activeSequence;
        if (!seq) return "ERROR: No active sequence";

        var track = seq.videoTracks[trackIndex];
        if (!track || clipIndex >= track.clips.numItems) return "ERROR: Invalid track or clip index";

        var clip = track.clips[clipIndex];
        var components = clip.components;
        
        for (var i = 0; i < components.numItems; i++) {
            var comp = components[i];
            for (var j = 0; j < comp.properties.numItems; j++) {
                var prop = comp.properties[j];
                if (prop.displayName === propertyName) {
                    if (prop.isTimeVarying()) {
                        var kfCount = prop.getKeys().length;
                        for (var k = 0; k < kfCount; k++) {
                            // 5 represents Bezier interpolation mode in Premiere DOM
                            prop.setInterpolationTypeAtKey(prop.getKeys()[k], 5, true);
                        }
                        return "SUCCESS: Interpolation updated to Bezier for " + propertyName;
                    }
                }
            }
        }
        return "WARN: Property " + propertyName + " not found or not time-varying";
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}

