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

          # Avoid runtime errors in numpy-dependent scripts when running them
          # in the host env by using Nix's native package management for Python.
          # <https://gist.github.com/GuillaumeDesforges/7d66cf0f63038724acf06f17331c9280>
          (pkgs.python312.withPackages (
            python-pkgs: with python-pkgs; [
              numpy
            ]
          ))
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
          ]
          ++ inputsForScripts
          ++ inputsTooling
          ++ inputsLsp;
        };

        # For compatibility with older versions of the `nix` binary.
        devShell = self.devShells.${system}.default;

        # Formatter to use with the `nix fmt` command.
        formatter = pkgs.nixfmt-tree;
      }
    );
}
