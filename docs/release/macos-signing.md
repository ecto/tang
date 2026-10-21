# macOS signing and notarization (one-time setup)

`.github/workflows/mac-release.yml` builds `tang-llm` for Apple Silicon, signs it with a
Developer ID, notarizes it, and attaches `tang-llm-aarch64-apple-darwin.tar.gz` to the release.
It needs six repository secrets. Only the Apple account holder can create them.

1. **Developer ID certificate.** In Xcode → Settings → Accounts → (team) → Manage Certificates →
   "+" → Developer ID Application. (Or developer.apple.com → Certificates → "+" → Developer ID
   Application, with a CSR from Keychain Access.)
2. **Export it.** Keychain Access → My Certificates → "Developer ID Application: …" → right-click →
   Export → `.p12` with a strong password. Then `base64 -i cert.p12 | pbcopy`.
3. **Notary API key.** appstoreconnect.apple.com → Users and Access → Integrations → App Store
   Connect API → Team Keys → "+" → name "notary", access "Developer". Download `AuthKey_<id>.p8`
   (only once). Note the Key ID and the Issuer ID shown above the table.
   `base64 -i AuthKey_<id>.p8 | pbcopy`.
4. **Secrets.** github.com/ecto/tang → Settings → Secrets and variables → Actions → New secret:
   - `MACOS_CERT_P12`: base64 from step 2
   - `MACOS_CERT_PASSWORD`: the .p12 password
   - `MACOS_SIGN_IDENTITY`: the certificate's full name, e.g.
     `Developer ID Application: Municipal Robotics (TEAMID1234)`
     (`security find-identity -v -p codesigning` prints it)
   - `ASC_KEY_ID`, `ASC_ISSUER_ID`: from step 3
   - `ASC_KEY_P8`: base64 from step 3
5. Delete the local `cert.p12` and `.p8` copies, or keep them in a password manager.

The same certificate and API key can sign other repos (for example vcad); add the same secrets there,
or as organization secrets on ecto to share them.

To release: push a `v*` tag, or run "mac release" by hand with an existing tag.
