{
  description = "Flake for development";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixos-25.05/nixexprs.tar.xz";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };

        inputsForScripts = [
          pkgs.jq
          pkgs.curl
        ];

        inputsTooling = [
          pkgs.uv
          pkgs.pre-commit

          # Totally optional.
          pkgs.lazydocker
        ];

        inputsLsp = [
          # Python.
          pkgs.pyright
          pkgs.ruff

          # Nix.
          pkgs.nixd
          pkgs.nixfmt-rfc-style
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.libGL
            pkgs.glib
          ]
          ++ inputsForScripts
          ++ inputsTooling
          ++ inputsLsp;

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.libGL}/lib:${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
          '';
        };

        # For compatibility with older versions of the `nix` binary.
        devShell = self.devShells.${system}.default;

        # Formatter to use with the `nix fmt` command.
        formatter = pkgs.nixfmt-tree;
      }
    );
}
