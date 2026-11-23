// Extends app.json with values that must not be committed.
// RNMAPBOX_MAPS_DOWNLOAD_TOKEN is only needed for native builds on older Mapbox SDKs.
module.exports = ({ config }) => {
  const token = process.env.RNMAPBOX_MAPS_DOWNLOAD_TOKEN;
  if (token) {
    config.plugins = config.plugins.map((p) =>
      Array.isArray(p) && p[0] === '@rnmapbox/maps' ? [p[0], { RNMapboxMapsDownloadToken: token }] : p,
    );
  }
  return config;
};
